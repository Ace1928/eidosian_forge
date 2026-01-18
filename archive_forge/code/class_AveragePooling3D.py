import warnings
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.legacy_tf_layers import base
class AveragePooling3D(keras_layers.AveragePooling3D, base.Layer):
    """Average pooling layer for 3D inputs (e.g. volumes).

  Args:
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.
  """

    def __init__(self, pool_size, strides, padding='valid', data_format='channels_last', name=None, **kwargs):
        if strides is None:
            raise ValueError('Argument `strides` must not be None.')
        super(AveragePooling3D, self).__init__(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, name=name, **kwargs)