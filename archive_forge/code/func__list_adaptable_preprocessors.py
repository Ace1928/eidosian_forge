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
def _list_adaptable_preprocessors(self):
    adaptable_preprocessors = []
    for name in self.features.keys():
        preprocessor = self.preprocessors[name]
        if isinstance(preprocessor, layers.Normalization):
            if preprocessor.input_mean is not None:
                continue
        if hasattr(preprocessor, 'adapt'):
            adaptable_preprocessors.append(name)
    return adaptable_preprocessors