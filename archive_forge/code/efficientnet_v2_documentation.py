import copy
import math
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
A placeholder method for backward compatibility.

    The preprocessing logic has been included in the EfficientNetV2 model
    implementation. Users are no longer required to call this method to
    normalize the input data. This method does nothing and only kept as a
    placeholder to align the API surface between old and new version of model.

    Args:
      x: A floating point `numpy.array` or a `tf.Tensor`.
      data_format: Optional data format of the image tensor/array. `None` means
        the global setting `tf.keras.backend.image_data_format()` is used
        (unless you changed it, it uses "channels_last").
        Defaults to `None`.

    Returns:
      Unchanged `numpy.array` or `tf.Tensor`.
    