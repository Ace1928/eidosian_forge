import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn.base_wrapper import Wrapper
from keras.src.utils import generic_utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
def _remove_timesteps(self, dims):
    dims = dims.as_list()
    return tf.TensorShape([dims[0]] + dims[2:])