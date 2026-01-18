import numpy as np
from onnx.reference.op_run import OpRun
class BatchNormalization_9(OpRun):

    def _run(self, x, scale, bias, mean, var, epsilon=None, momentum=None):
        if momentum is None:
            res = _batchnorm_test_mode(x, scale, bias, mean, var, epsilon=epsilon)
            return (res,)
        axis = tuple(np.delete(np.arange(len(x.shape)), 1))
        saved_mean = x.mean(axis=axis)
        saved_var = x.var(axis=axis)
        output_mean = mean * momentum + saved_mean * (1 - momentum)
        output_var = var * momentum + saved_var * (1 - momentum)
        res = _batchnorm_test_mode(x, scale, bias, output_mean, output_var, epsilon=epsilon)
        return (res,)