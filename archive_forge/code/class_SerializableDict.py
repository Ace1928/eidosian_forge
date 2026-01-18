import importlib
import inspect
import types
import warnings
import numpy as np
from keras.src import api_export
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.saving import object_registration
from keras.src.utils import python_utils
from keras.src.utils.module_utils import tensorflow as tf
class SerializableDict:

    def __init__(self, **config):
        self.config = config

    def serialize(self):
        return serialize_keras_object(self.config)