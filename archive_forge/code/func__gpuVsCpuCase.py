from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
def _gpuVsCpuCase(self, in_shape, out_shape, align_corners, half_pixel_centers, dtype):
    grad = {}
    for use_gpu in [False, True]:
        grad[use_gpu] = self._getJacobians(in_shape, out_shape, align_corners, half_pixel_centers, dtype=dtype, use_gpu=use_gpu)
    threshold = 0.0001
    self.assertAllClose(grad[False], grad[True], rtol=threshold, atol=threshold)