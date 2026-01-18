import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import utils
from keras.src.applications import imagenet_utils
from keras.src.engine import sequential
from keras.src.engine import training as training_lib
from tensorflow.python.util.tf_export import keras_export
def Head(num_classes=1000, classifier_activation=None, name=None):
    """Implementation of classification head of ConvNeXt.

    Args:
      num_classes: number of classes for Dense layer
      classifier_activation: activation function for the Dense layer
      name: name prefix

    Returns:
      Classification head function.
    """
    if name is None:
        name = str(backend.get_uid('head'))

    def apply(x):
        x = layers.GlobalAveragePooling2D(name=name + '_head_gap')(x)
        x = layers.LayerNormalization(epsilon=1e-06, name=name + '_head_layernorm')(x)
        x = layers.Dense(num_classes, activation=classifier_activation, name=name + '_head_dense')(x)
        return x
    return apply