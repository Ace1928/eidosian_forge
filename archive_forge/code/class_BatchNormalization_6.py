import numpy as np
from onnx.reference.op_run import OpRun
class BatchNormalization_6(OpRun):

    def _run(self, x, scale, bias, mean, var, epsilon=None, is_test=None, momentum=None, spatial=None):
        if is_test:
            res = _batchnorm_test_mode(x, scale, bias, mean, var, epsilon=epsilon)
        else:
            res = _batchnorm_training_mode(x, scale, bias, mean, var, epsilon=epsilon, momentum=momentum)
        return (res,)