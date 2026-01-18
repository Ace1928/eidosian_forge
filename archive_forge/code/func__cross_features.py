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
def _cross_features(self, features):
    all_outputs = {}
    for cross in self.crosses:
        inputs = [features[name] for name in cross.feature_names]
        outputs = self.crossers[cross.name](inputs)
        all_outputs[cross.name] = outputs
    return all_outputs