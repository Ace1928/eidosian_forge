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
def _Check3DImage(image, require_static=True):
    """Assert that we are working with a properly shaped image.

  Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are known
      and non-zero.

  Raises:
    ValueError: if `image.shape` is not a 3-vector.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' (shape %s) must be three-dimensional." % image.shape)
    if require_static and (not image_shape.is_fully_defined()):
        raise ValueError("'image' (shape %s) must be fully defined." % image_shape)
    if any((x == 0 for x in image_shape)):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" % image_shape)
    if not image_shape.is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image), ["all dims of 'image.shape' must be > 0."])]
    else:
        return []