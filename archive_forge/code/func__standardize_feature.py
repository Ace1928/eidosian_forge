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
def _standardize_feature(self, name, feature):
    if isinstance(feature, Feature):
        return feature
    if isinstance(feature, dict):
        return serialization_lib.deserialize_keras_object(feature)
    if feature == 'float':
        return self.float(name=name)
    elif feature == 'float_normalized':
        return self.float_normalized(name=name)
    elif feature == 'float_rescaled':
        return self.float_rescaled(name=name)
    elif feature == 'float_discretized':
        return self.float_discretized(name=name, num_bins=self.num_discretization_bins)
    elif feature == 'integer_categorical':
        return self.integer_categorical(name=name)
    elif feature == 'string_categorical':
        return self.string_categorical(name=name)
    elif feature == 'integer_hashed':
        return self.integer_hashed(self.hashing_dim, name=name)
    elif feature == 'string_hashed':
        return self.string_hashed(self.hashing_dim, name=name)
    else:
        raise ValueError(f'Invalid feature type: {feature}')