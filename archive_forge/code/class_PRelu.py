import numpy as np
from onnx.reference.op_run import OpRun
class PRelu(OpRun):

    def _run(self, x, slope):
        try:
            return (np.where(x > 0, x, x * slope).astype(x.dtype),)
        except ValueError as e:
            if len(slope.shape) == 1:
                dim = slope.shape[0]
                new_shape = []
                n = 0
                for d in x.shape:
                    if d == dim:
                        new_shape.append(d)
                        n += 1
                    else:
                        new_shape.append(1)
                if n == 1:
                    xs = x * slope.reshape(tuple(new_shape))
                    return (np.where(x > 0, x, xs).astype(x.dtype),)
            raise e