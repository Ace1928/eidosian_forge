import re
import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export
Analyze an pre-split einsum string to find the weight shape.