import warnings
from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.keras.legacy_tf_layers import base
class AveragePooling1D(keras_layers.AveragePooling1D, base.Layer):
    """Average Pooling layer for 1D inputs.

  Args:
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    name: A string, the name of the layer.
  """

    def __init__(self, pool_size, strides, padding='valid', data_format='channels_last', name=None, **kwargs):
        if strides is None:
            raise ValueError('Argument `strides` must not be None.')
        super(AveragePooling1D, self).__init__(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, name=name, **kwargs)