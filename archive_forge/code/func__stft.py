import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_concat_from_sequence import _concat_from_sequence
from onnx.reference.ops.op_dft import _cfft as _dft
from onnx.reference.ops.op_slice import _slice
def _stft(x, fft_length: int, hop_length, n_frames, window, onesided=False):
    """Applies one dimensional FFT with window weights.

    torch defines the number of frames as:
    `n_frames = 1 + (len - n_fft) // hop_length`.
    """
    last_axis = len(x.shape) - 1
    axis = [-2]
    axis2 = [-3]
    window_size = window.shape[0]
    seq = []
    for fs in range(n_frames):
        begin = fs * hop_length
        end = begin + window_size
        sliced_x = _slice(x, np.array([begin]), np.array([end]), axis)
        new_dim = sliced_x.shape[-2:-1]
        missing = (window_size - new_dim[0],)
        new_shape = sliced_x.shape[:-2] + missing + sliced_x.shape[-1:]
        cst = np.zeros(new_shape, dtype=x.dtype)
        pad_sliced_x = _concat(sliced_x, cst, axis=-2)
        un_sliced_x = _unsqueeze(pad_sliced_x, axis2)
        seq.append(un_sliced_x)
    new_x = _concat_from_sequence(seq, axis=-3, new_axis=0)
    shape_x = new_x.shape
    shape_x_short = shape_x[:-2]
    shape_x_short_one = tuple((1 for _ in shape_x_short))
    window_shape = (*shape_x_short_one, window_size, 1)
    weights = np.reshape(window, window_shape)
    weighted_new_x = new_x * weights
    result = _dft(weighted_new_x, fft_length, last_axis, onesided=onesided, normalize=False)
    return result