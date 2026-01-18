from ._internal import NDArrayBase
from ..base import _Null
def RNN(data=None, parameters=None, state=None, state_cell=None, sequence_length=None, state_size=_Null, num_layers=_Null, bidirectional=_Null, mode=_Null, p=_Null, state_outputs=_Null, projection_size=_Null, lstm_state_clip_min=_Null, lstm_state_clip_max=_Null, lstm_state_clip_nan=_Null, use_sequence_length=_Null, out=None, name=None, **kwargs):
    """Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are
    implemented, with both multi-layer and bidirectional support.

    When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
    and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
    pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
    Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

    **Vanilla RNN**

    Applies a single-gate recurrent layer to input X. Two kinds of activation function are supported:
    ReLU and Tanh.

    With ReLU activation function:

    .. math::
        h_t = relu(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

    With Tanh activtion function:

    .. math::
        h_t = \\tanh(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

    Reference paper: Finding structure in time - Elman, 1988.
    https://crl.ucsd.edu/~elman/Papers/fsit.pdf

    **LSTM**

    Long Short-Term Memory - Hochreiter, 1997. http://www.bioinf.jku.at/publications/older/2604.pdf

    .. math::
      \\begin{array}{ll}
                i_t = \\mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\\\
                f_t = \\mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\\\
                g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\\\
                o_t = \\mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\\\
                c_t = f_t * c_{(t-1)} + i_t * g_t \\\\
                h_t = o_t * \\tanh(c_t)
                \\end{array}

    With the projection size being set, LSTM could use the projection feature to reduce the parameters
    size and give some speedups without significant damage to the accuracy.

    Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech
    Recognition - Sak et al. 2014. https://arxiv.org/abs/1402.1128

    .. math::
      \\begin{array}{ll}
                i_t = \\mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{ri} r_{(t-1)} + b_{ri}) \\\\
                f_t = \\mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{rf} r_{(t-1)} + b_{rf}) \\\\
                g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{rc} r_{(t-1)} + b_{rg}) \\\\
                o_t = \\mathrm{sigmoid}(W_{io} x_t + b_{o} + W_{ro} r_{(t-1)} + b_{ro}) \\\\
                c_t = f_t * c_{(t-1)} + i_t * g_t \\\\
                h_t = o_t * \\tanh(c_t)
                r_t = W_{hr} h_t
                \\end{array}

    **GRU**

    Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078

    The definition of GRU here is slightly different from paper but compatible with CUDNN.

    .. math::
      \\begin{array}{ll}
                r_t = \\mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\\\
                z_t = \\mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\\\
                n_t = \\tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\\\
                h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\\\
                \\end{array}


    Defined in ../src/operator/rnn.cc:L375

    Parameters
    ----------
    data : NDArray
        Input data to RNN
    parameters : NDArray
        Vector of all RNN trainable parameters concatenated
    state : NDArray
        initial hidden state of the RNN
    state_cell : NDArray
        initial cell state for LSTM networks (only for LSTM)
    sequence_length : NDArray
        Vector of valid sequence lengths for each element in batch. (Only used if use_sequence_length kwarg is True)
    state_size : int (non-negative), required
        size of the state for each layer
    num_layers : int (non-negative), required
        number of stacked layers
    bidirectional : boolean, optional, default=0
        whether to use bidirectional recurrent layers
    mode : {'gru', 'lstm', 'rnn_relu', 'rnn_tanh'}, required
        the type of RNN to compute
    p : float, optional, default=0
        drop rate of the dropout on the outputs of each RNN layer, except the last layer.
    state_outputs : boolean, optional, default=0
        Whether to have the states as symbol outputs.
    projection_size : int or None, optional, default='None'
        size of project size
    lstm_state_clip_min : double or None, optional, default=None
        Minimum clip value of LSTM states. This option must be used together with lstm_state_clip_max.
    lstm_state_clip_max : double or None, optional, default=None
        Maximum clip value of LSTM states. This option must be used together with lstm_state_clip_min.
    lstm_state_clip_nan : boolean, optional, default=0
        Whether to stop NaN from propagating in state by clipping it to min/max. If clipping range is not specified, this option is ignored.
    use_sequence_length : boolean, optional, default=0
        If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)