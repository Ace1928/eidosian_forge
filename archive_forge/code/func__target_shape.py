from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.feature_column import base_feature_layer as kfc
from keras.src.saving.legacy.saved_model import json_utils
from tensorflow.python.util.tf_export import keras_export
def _target_shape(self, input_shape, total_elements):
    return (input_shape[0], total_elements)