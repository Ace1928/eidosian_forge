import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine import base_layer
from keras.src.layers.rnn.base_conv_rnn import ConvRNN
from keras.src.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.src.utils import conv_utils
class ConvLSTM(ConvRNN):
    """Abstract N-D Convolutional LSTM layer (used as implementation base).

    Similar to an LSTM layer, but the input transformations
    and recurrent transformations are both convolutional.

    Args:
      rank: Integer, rank of the convolution, e.g. "2" for 2D convolutions.
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, time, ..., channels)`
        while `channels_first` corresponds to
        inputs with shape `(batch, time, channels, ...)`.
        When unspecified, uses
        `image_data_format` value found in your Keras config file at
         `~/.keras/keras.json` (if exists) else 'channels_last'.
        Defaults to 'channels_last'.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
        By default hyperbolic tangent activation function is applied
        (`tanh(x)`).
      recurrent_activation: Activation function to use
        for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Use in combination with `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et al., 2015](
          http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to.
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence. (default False)
      return_state: Boolean Whether to return the last state
        in addition to the output. (default False)
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
    """

    def __init__(self, rank, filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0, **kwargs):
        cell = ConvLSTMCell(rank=rank, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, dropout=dropout, recurrent_dropout=recurrent_dropout, name='conv_lstm_cell', dtype=kwargs.get('dtype'))
        super().__init__(rank, cell, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters, 'kernel_size': self.kernel_size, 'strides': self.strides, 'padding': self.padding, 'data_format': self.data_format, 'dilation_rate': self.dilation_rate, 'activation': activations.serialize(self.activation), 'recurrent_activation': activations.serialize(self.recurrent_activation), 'use_bias': self.use_bias, 'kernel_initializer': initializers.serialize(self.kernel_initializer), 'recurrent_initializer': initializers.serialize(self.recurrent_initializer), 'bias_initializer': initializers.serialize(self.bias_initializer), 'unit_forget_bias': self.unit_forget_bias, 'kernel_regularizer': regularizers.serialize(self.kernel_regularizer), 'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer), 'bias_regularizer': regularizers.serialize(self.bias_regularizer), 'activity_regularizer': regularizers.serialize(self.activity_regularizer), 'kernel_constraint': constraints.serialize(self.kernel_constraint), 'recurrent_constraint': constraints.serialize(self.recurrent_constraint), 'bias_constraint': constraints.serialize(self.bias_constraint), 'dropout': self.dropout, 'recurrent_dropout': self.recurrent_dropout}
        base_config = super().get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)