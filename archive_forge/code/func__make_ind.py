import numpy as np
from onnx.reference.op_run import OpRun
def _make_ind(dim, shape):
    m = np.empty(shape, dtype=np.int64)
    ind = [slice(0, shape[i]) for i in range(len(shape))]
    new_shape = [1] * len(shape)
    new_shape[dim] = shape[dim]
    first = np.arange(shape[dim]).reshape(new_shape)
    m[tuple(ind)] = first
    return m