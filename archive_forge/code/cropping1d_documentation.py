import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import conv_utils
from tensorflow.python.util.tf_export import keras_export
Cropping layer for 1D input (e.g. temporal sequence).

    It crops along the time dimension (axis 1).

    Examples:

    >>> input_shape = (2, 3, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[ 0  1]
      [ 2  3]
      [ 4  5]]
     [[ 6  7]
      [ 8  9]
      [10 11]]]
    >>> y = tf.keras.layers.Cropping1D(cropping=1)(x)
    >>> print(y)
    tf.Tensor(
      [[[2 3]]
       [[8 9]]], shape=(2, 1, 2), dtype=int64)

    Args:
      cropping: Int or tuple of int (length 2)
        How many units should be trimmed off at the beginning and end of
        the cropping dimension (axis 1).
        If a single int is provided, the same value will be used for both.

    Input shape:
      3D tensor with shape `(batch_size, axis_to_crop, features)`

    Output shape:
      3D tensor with shape `(batch_size, cropped_axis, features)`
    