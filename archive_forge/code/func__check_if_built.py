import tree
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.utils import backend_utils
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
def _check_if_built(self):
    if not self._sublayers_built:
        self._check_if_adapted()
        self.get_encoded_features()
        self._sublayers_built = True