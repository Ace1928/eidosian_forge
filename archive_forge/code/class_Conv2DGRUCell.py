from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
class Conv2DGRUCell(_ConvGRUCell):
    """2D Convolutional Gated Rectified Unit (GRU) network cell.

    .. math::
        \\begin{array}{ll}
        r_t = \\sigma(W_r \\ast x_t + R_r \\ast h_{t-1} + b_r) \\\\
        z_t = \\sigma(W_z \\ast x_t + R_z \\ast h_{t-1} + b_z) \\\\
        n_t = tanh(W_i \\ast x_t + b_i + r_t \\circ (R_n \\ast h_{t-1} + b_n)) \\\\
        h^\\prime_t = (1 - z_t) \\circ n_t + z_t \\circ h \\\\
        \\end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCHW' the shape should be (C, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCHW' and 'NHWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function used in n_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    prefix : str, default ``'conv_gru_``'
        Prefix for name of layers (and name of weight if params is None).
    params : RNNParams, default None
        Container for weight sharing between cells. Created if None.
    """

    def __init__(self, input_shape, hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, 0), i2h_dilate=(1, 1), h2h_dilate=(1, 1), i2h_weight_initializer=None, h2h_weight_initializer=None, i2h_bias_initializer='zeros', h2h_bias_initializer='zeros', conv_layout='NCHW', activation='tanh', prefix=None, params=None):
        super(Conv2DGRUCell, self).__init__(input_shape=input_shape, hidden_channels=hidden_channels, i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel, i2h_pad=i2h_pad, i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate, i2h_weight_initializer=i2h_weight_initializer, h2h_weight_initializer=h2h_weight_initializer, i2h_bias_initializer=i2h_bias_initializer, h2h_bias_initializer=h2h_bias_initializer, dims=2, conv_layout=conv_layout, activation=activation, prefix=prefix, params=params)