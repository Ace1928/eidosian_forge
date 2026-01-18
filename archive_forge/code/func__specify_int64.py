import numpy as np
from onnx.reference.op_run import OpRun
def _specify_int64(indices, inverse_indices, counts):
    return (np.array(indices, dtype=np.int64), np.array(inverse_indices, dtype=np.int64), np.array(counts, dtype=np.int64))