import math
import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import utils
from keras.src.saving import serialization_lib
from tensorflow.python.util.tf_export import keras_export
def _ensure_keras_seeded():
    """Make sure the keras.backend global seed generator is set.

    This is important for DTensor use case to ensure that each client are
    initialized with same seed for tf.random.Generator, so that the value
    created are in sync among all the clients.
    """
    if not getattr(backend._SEED_GENERATOR, 'generator', None):
        raise ValueError('When using DTensor APIs, you need to set the global seed before using any Keras initializers. Please make sure to call `tf.keras.utils.set_random_seed()` in your code.')