import numpy as np
from onnx.reference.op_run import OpRun
class Unsqueeze_1(OpRun):

    def _run(self, data, axes=None):
        if isinstance(axes, np.ndarray):
            axes = tuple(axes)
        elif axes in ([], ()):
            axes = None
        elif isinstance(axes, list):
            axes = tuple(axes)
        if isinstance(axes, (tuple, list)):
            sq = data
            for a in axes:
                sq = np.expand_dims(sq, axis=a)
        else:
            raise RuntimeError('axes cannot be None for operator Unsqueeze (Unsqueeze_1).')
        return (sq,)