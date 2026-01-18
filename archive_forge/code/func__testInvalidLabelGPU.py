import numpy as np
from tensorflow.python.eager import backprop as backprop_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
@test_util.run_gpu_only()
def _testInvalidLabelGPU(self, invalid_label_gradient=np.nan):
    labels = [4, 3, 0, -1]
    logits = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    loss, gradient = self._opFwdBwd(labels=labels, logits=logits)
    self.assertAllClose([np.nan, 1.3862, 3.442, np.nan], loss, rtol=0.001, atol=0.001)
    self.assertAllClose([[invalid_label_gradient] * 4, [0.25, 0.25, 0.25, -0.75], [-0.968, 0.087, 0.237, 0.6439], [invalid_label_gradient] * 4], gradient, rtol=0.001, atol=0.001)