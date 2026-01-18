import tensorflow.compat.v2 as tf
from keras.src.layers.pooling.base_pooling3d import Pooling3D
from tensorflow.python.util.tf_export import keras_export
Average pooling operation for 3D data (spatial or spatio-temporal).

    Downsamples the input along its spatial dimensions (depth, height, and
    width) by taking the average value over an input window
    (of size defined by `pool_size`) for each channel of the input.
    The window is shifted by `strides` along each dimension.

    Args:
      pool_size: tuple of 3 integers,
        factors by which to downscale (dim1, dim2, dim3).
        `(2, 2, 2)` will halve the size of the 3D input in each dimension.
      strides: tuple of 3 integers, or None. Strides values.
      padding: One of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        while `channels_first` corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
        When unspecified, uses
        `image_data_format` value found in your Keras config file at
         `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to 'channels_last'.

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`

    Example:

    ```python
    depth = 30
    height = 30
    width = 30
    input_channels = 3

    inputs = tf.keras.Input(shape=(depth, height, width, input_channels))
    layer = tf.keras.layers.AveragePooling3D(pool_size=3)
    outputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)
    ```
    