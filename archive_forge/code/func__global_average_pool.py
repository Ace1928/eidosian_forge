import numpy as np
from onnx.reference.op_run import OpRun
def _global_average_pool(x: np.ndarray) -> np.ndarray:
    axis = tuple(range(2, np.ndim(x)))
    y = np.average(x, axis=axis)
    for _ in axis:
        y = np.expand_dims(y, -1)
    return y