import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.distribute import model_collection_base
from keras.src.optimizers.legacy import gradient_descent
class SimpleTFModuleModel(model_collection_base.ModelAndInput):
    """A simple model based on tf.Module and its data."""

    def get_model(self, **kwargs):
        model = _SimpleModule()
        return model

    def get_data(self):
        return _get_data_for_simple_models()

    def get_batch_size(self):
        return _BATCH_SIZE