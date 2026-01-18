from keras.src.layers.rnn.base_conv_lstm import ConvLSTM
from tensorflow.python.util.tf_export import keras_export
1D Convolutional LSTM.

    Similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        dimensions of the convolution window.
      strides: An integer or tuple/list of n integers, specifying the strides of
        the convolution. Specifying any stride value != 1 is incompatible with
        specifying any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive). `"valid"` means
        no padding. `"same"` results in padding evenly to the left/right or
        up/down of the input such that output has the same height/width
        dimension as the input.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.  The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, time, ...,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, time, channels, ...)`. When unspecified, uses
        `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to 'channels_last'.
      dilation_rate: An integer or tuple/list of n integers, specifying the
        dilation rate to use for dilated convolution. Currently, specifying any
        `dilation_rate` value != 1 is incompatible with specifying any `strides`
        value != 1.
      activation: Activation function to use. By default hyperbolic tangent
        activation function is applied (`tanh(x)`).
      recurrent_activation: Activation function to use for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate
        at initialization. Use in combination with `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et al., 2015](
        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output in the output
        sequence, or the full sequence. (default False)
      return_state: Boolean Whether to return the last state in addition to the
        output. (default False)
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards.
      stateful: Boolean (default False). If True, the last state for each sample
        at index i in a batch will be used as initial state for the sample of
        index i in the following batch.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
        for the linear transformation of the recurrent state.
    Call arguments:
      inputs: A 4D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether a
        given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` are set.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
    Input shape: - If data_format='channels_first'
          4D tensor with shape: `(samples, time, channels, rows)` - If
            data_format='channels_last'
          4D tensor with shape: `(samples, time, rows, channels)`
    Output shape:
      - If `return_state`: a list of tensors. The first tensor is the output.
        The remaining tensors are the last states,
        each 3D tensor with shape: `(samples, filters, new_rows)` if
          data_format='channels_first'
        or shape: `(samples, new_rows, filters)` if data_format='channels_last'.
          `rows` values might have changed due to padding.
      - If `return_sequences`: 4D tensor with shape: `(samples, timesteps,
        filters, new_rows)` if data_format='channels_first'
        or shape: `(samples, timesteps, new_rows, filters)` if
          data_format='channels_last'.
      - Else, 3D tensor with shape: `(samples, filters, new_rows)` if
        data_format='channels_first'
        or shape: `(samples, new_rows, filters)` if data_format='channels_last'.

    Raises:
      ValueError: in case of invalid constructor arguments.

    References:
      - [Shi et al., 2015](http://arxiv.org/abs/1506.04214v1)
      (the current implementation does not include the feedback loop on the
      cells output).
    