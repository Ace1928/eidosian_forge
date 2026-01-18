import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.distribute import model_collection_base
from keras.src.optimizers.legacy import gradient_descent
class SimpleSubclassModel(model_collection_base.ModelAndInput):
    """A simple subclass model and its data."""

    def get_model(self, **kwargs):
        model = _SimpleModel()
        optimizer = gradient_descent.SGD(learning_rate=0.001)
        model.compile(loss='mse', metrics=['mae'], cloning=False, optimizer=optimizer)
        return model

    def get_data(self):
        return _get_data_for_simple_models()

    def get_batch_size(self):
        return _BATCH_SIZE