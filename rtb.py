import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

import pandas as pd
data = pd.read_csv("../input/rtb.csv")
print(data.shape)
data.head()

import seaborn as sns
sns.countplot(data.convert)
data.convert.value_counts()

count_classes = pd.value_counts(data['convert'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("bidding conversion histogram")
plt.xlabel("Conversion")
plt.ylabel("Count")

train = data[:800000]
test = data[800000:]

from sklearn.model_selection import train_test_split
train, test=train_test_split(data, test_size=0.2, stratify=data.convert )

train[train.convert==1]
test[test.convert==1]

def undersample(data, ratio=1):
    conv = data[data.convert == 1]
    oth = data[data.convert == 0].sample(n=ratio*len(conv))
    return pd.concat([conv, oth]).sample(frac=1) #shuffle data

ustrain = undersample(train)

y_train = ustrain.convert
X_train = ustrain.drop('convert', axis=1)

print("Remaining rows", len(ustrain))

from imblearn.keras import balanced_batch_generator

ustrain = undersample(test)
y_test = ustrain.convert
X_test = ustrain.drop('convert', axis=1)

print("Remaining rows", len(ustrain))

from keras import models
from keras import layers
from keras import regularizers
model = models.Sequential()
# kernel_regularizer=regularizers.l2(0.001),

model.add(layers.Dense(12, 
                       activation='relu', input_shape=(88,)))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(12, 
                       activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Recall, F1, Precision
from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(optimizer='adam',
              loss='binary_crossentropy',              
              metrics=[f1_m, precision_m, recall_m])

from keras import optimizers

model.compile(optimizer=optimizers.Adam(lr=0.01),
             loss='binary_crossentropy',
             metrics=[f1_m, precision_m, recall_m])

from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.Adam(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[f1_m])

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(X_train,
                   y_train,
                   epochs=30,
                   batch_size=50, callbacks = [es], validation_split=0.2)

import matplotlib.pyplot as plt

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Traning loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()
f1_m = history_dict['f1_m']
val_f1_m = history_dict['val_f1_m']

plt.plot(epochs, f1_m, 'bo', label='Training f1_m')
plt.plot(epochs, val_f1_m, 'b', label='Validation f1_m')
plt.title('Training and Validation f1_m')
plt.xlabel('Epochs')
plt.ylabel('f1_m')
plt.legend()

plt.show()

from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from keras import regularizers
model = models.Sequential()
# kernel_regularizer=regularizers.l2(0.001),

model.add(layers.Dense(12,
                       activation='relu', input_shape=(88,)))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(12, 
                       activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=[f1_m])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=2)

model.fit(X_train, y_train, epochs=30, callbacks = [es], batch_size=128, validation_data=(X_test, y_test))
# results = model.evaluate(X_test, y_test)

results
model.predict(X_test)