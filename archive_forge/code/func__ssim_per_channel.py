import functools
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _ssim_per_channel(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_index_map=False):
    """Computes SSIM index between img1 and img2 per color channel.

  This function matches the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.

  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.

  Args:
    img1: First image batch.
    img2: Second image batch.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Default value 11 (size of gaussian filter).
    filter_sigma: Default value 1.5 (width of gaussian filter).
    k1: Default value 0.01
    k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
      it would be better if we took the values in the range of 0 < K2 < 0.4).
    return_index_map: If True returns local SSIM map instead of the global mean.

  Returns:
    A pair of tensors containing and channel-wise SSIM and contrast-structure
    values. The shape is [..., channels].
  """
    filter_size = constant_op.constant(filter_size, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(filter_sigma, dtype=img1.dtype)
    shape1, shape2 = array_ops.shape_n([img1, img2])
    checks = [control_flow_assert.Assert(math_ops.reduce_all(math_ops.greater_equal(shape1[-3:-1], filter_size)), [shape1, filter_size], summarize=8), control_flow_assert.Assert(math_ops.reduce_all(math_ops.greater_equal(shape2[-3:-1], filter_size)), [shape2, filter_size], summarize=8)]
    with ops.control_dependencies(checks):
        img1 = array_ops.identity(img1)
    kernel = _fspecial_gauss(filter_size, filter_sigma)
    kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])
    compensation = 1.0

    def reducer(x):
        shape = array_ops.shape(x)
        x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
        y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return array_ops.reshape(y, array_ops.concat([shape[:-3], array_ops.shape(y)[1:]], 0))
    luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation, k1, k2)
    if return_index_map:
        ssim_val = luminance * cs
    else:
        axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
        ssim_val = math_ops.reduce_mean(luminance * cs, axes)
        cs = math_ops.reduce_mean(cs, axes)
    return (ssim_val, cs)