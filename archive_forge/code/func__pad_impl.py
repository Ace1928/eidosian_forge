import numpy as np
from onnx.reference.op_run import OpRun
def _pad_impl(data, raw_pads, mode, constant_values=0.0, axes=None):
    input_rank = data.ndim
    if axes is None:
        axes = list(range(input_rank))
    else:
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
    num_axes = len(axes)
    if num_axes * 2 != len(raw_pads):
        raise RuntimeError('The number of elements in raw_pads should be 2 times the number of axes')
    pad_width = [(0, 0)] * input_rank
    for i, axis in enumerate(axes):
        pad_begin = raw_pads[i]
        pad_end = raw_pads[num_axes + i]
        pad_width[axis] = (pad_begin, pad_end)
    if mode == 'constant':
        return np.pad(data, pad_width=pad_width, mode=mode, constant_values=constant_values).astype(data.dtype)
    return np.pad(data, pad_width=pad_width, mode=mode).astype(data.dtype)