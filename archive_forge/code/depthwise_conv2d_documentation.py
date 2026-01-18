from keras.src import backend
from keras.src.layers.convolutional.base_depthwise_conv import DepthwiseConv
from keras.src.utils import conv_utils
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Depthwise 2D convolution.

    Depthwise convolution is a type of convolution in which each input channel
    is convolved with a different kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable
    convolution.

    It is implemented via the following steps:

    - Split the input into individual channels.
    - Convolve each channel with an individual depthwise kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.

    Unlike a regular 2D convolution, depthwise convolution does not mix
    information across different input channels.

    The `depth_multiplier` argument determines how many filter are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.

    Args:
      kernel_size: An integer or tuple/list of 2 integers, specifying the height
        and width of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width. Can be a single integer to
        specify the same value for all spatial dimensions. Current
        implementation only supports equal length strides in row and
        column dimensions. Specifying any stride value != 1 is incompatible
        with specifying any `dilation_rate` value !=1.
      padding: one of `'valid'` or `'same'` (case-insensitive). `"valid"` means
        no padding. `"same"` results in padding with zeros evenly to the
        left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      depth_multiplier: The number of depthwise convolution output channels for
        each input channel. The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch_size, height,
        width, channels)` while `channels_first` corresponds to inputs with
        shape `(batch_size, channels, height, width)`. When unspecified, uses
        `image_data_format` value found in your Keras config file at
         `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to 'channels_last'.
      dilation_rate: An integer or tuple/list of 2 integers, specifying the
        dilation rate to use for dilated convolution. Currently, specifying any
        `dilation_rate` value != 1 is incompatible with specifying any `strides`
        value != 1.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      depthwise_initializer: Initializer for the depthwise kernel matrix (see
        `keras.initializers`). If None, the default initializer
        ('glorot_uniform') will be used.
      bias_initializer: Initializer for the bias vector (see
        `keras.initializers`). If None, the default initializer ('zeros') will
        be used.
      depthwise_regularizer: Regularizer function applied to the depthwise
        kernel matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector (see
        `keras.regularizers`).
      activity_regularizer: Regularizer function applied to the output of the
        layer (its 'activation') (see `keras.regularizers`).
      depthwise_constraint: Constraint function applied to the depthwise kernel
        matrix (see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector (see
        `keras.constraints`).

    Input shape:
      4D tensor with shape: `[batch_size, channels, rows, cols]` if
        data_format='channels_first'
      or 4D tensor with shape: `[batch_size, rows, cols, channels]` if
        data_format='channels_last'.

    Output shape:
      4D tensor with shape: `[batch_size, channels * depth_multiplier, new_rows,
        new_cols]` if `data_format='channels_first'`
        or 4D tensor with shape: `[batch_size,
        new_rows, new_cols, channels * depth_multiplier]` if
        `data_format='channels_last'`. `rows` and `cols` values might have
        changed due to padding.

    Returns:
      A tensor of rank 4 representing
      `activation(depthwiseconv2d(inputs, kernel) + bias)`.

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.
    