import numpy as np
from onnx.reference.op_run import OpRun
class Unsqueeze_13(OpRun):

    def _run(self, data, axes=None):
        if axes is not None:
            if hasattr(axes, '__iter__') and len(axes.shape) > 0:
                try:
                    sq = np.expand_dims(data, axis=tuple(axes))
                except TypeError:
                    if len(axes) == 1:
                        sq = np.expand_dims(data, axis=tuple(axes)[0])
                    else:
                        sq = data
                        for a in reversed(axes):
                            sq = np.expand_dims(sq, axis=a)
            else:
                sq = np.expand_dims(data, axis=axes)
        else:
            raise RuntimeError('axes cannot be None for operator Unsqueeze (Unsqueeze_13).')
        return (sq,)