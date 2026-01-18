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
def _itGen(self, smaller_shape, larger_shape):
    up_sample = (smaller_shape, larger_shape)
    down_sample = (larger_shape, smaller_shape)
    pass_through = (larger_shape, larger_shape)
    shape_pairs = (up_sample, down_sample, pass_through)
    options = [(True, False)]
    if not test_util.is_xla_enabled():
        options += [(False, True), (False, False)]
    for align_corners, half_pixel_centers in options:
        for in_shape, out_shape in shape_pairs:
            yield (in_shape, out_shape, align_corners, half_pixel_centers)