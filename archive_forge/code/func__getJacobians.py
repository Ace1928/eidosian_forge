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
def _getJacobians(self, in_shape, out_shape, align_corners=False, half_pixel_centers=False, dtype=np.float32, use_gpu=False, force_gpu=False):
    with self.cached_session(use_gpu=use_gpu, force_gpu=force_gpu):
        x = np.arange(np.prod(in_shape)).reshape(in_shape).astype(dtype)
        input_tensor = constant_op.constant(x, shape=in_shape)

        def func(in_tensor):
            return image_ops.resize_bilinear(in_tensor, out_shape[1:3], align_corners=align_corners, half_pixel_centers=half_pixel_centers)
        return gradient_checker_v2.compute_gradient(func, [input_tensor])