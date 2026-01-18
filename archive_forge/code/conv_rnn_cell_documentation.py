from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
3D Convolutional Gated Rectified Unit (GRU) network cell.

    .. math::
        \begin{array}{ll}
        r_t = \sigma(W_r \ast x_t + R_r \ast h_{t-1} + b_r) \\
        z_t = \sigma(W_z \ast x_t + R_z \ast h_{t-1} + b_z) \\
        n_t = tanh(W_i \ast x_t + b_i + r_t \circ (R_n \ast h_{t-1} + b_n)) \\
        h^\prime_t = (1 - z_t) \circ n_t + z_t \circ h \\
        \end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCDHW' the shape should be (C, D, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCDHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCDHW' and 'NDHWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function used in n_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    prefix : str, default ``'conv_gru_``'
        Prefix for name of layers (and name of weight if params is None).
    params : RNNParams, default None
        Container for weight sharing between cells. Created if None.
    