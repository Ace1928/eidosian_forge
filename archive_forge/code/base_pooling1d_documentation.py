import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import conv_utils
Pooling layer for arbitrary pooling functions, for 1D inputs.

    This class only exists for code reuse. It will never be an exposed API.

    Args:
      pool_function: The pooling function to apply, e.g. `tf.nn.max_pool2d`.
      pool_size: An integer or tuple/list of a single integer,
        representing the size of the pooling window.
      strides: An integer or tuple/list of a single integer, specifying the
        strides of the pooling operation.
      padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first`
        corresponds to inputs with shape
        `(batch, features, steps)`.
      name: A string, the name of the layer.
    