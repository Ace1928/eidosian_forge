import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import conv_utils
Calculates padding for 'causal' option for 1-d conv layers.