import numpy as np
from absl import app
from absl import flags
from absl import logging
import keras.src as keras
@memory_profiler.profile
def _imdb_lstm_model():
    """LSTM model."""
    x_train = np.random.randint(0, 1999, size=(2500, 100))
    y_train = np.random.random((2500, 1))
    model = keras.Sequential()
    model.add(keras.layers.Embedding(20000, 128))
    model.add(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile('sgd', 'mse')
    model.fit(x_train, y_train, batch_size=512, epochs=3)