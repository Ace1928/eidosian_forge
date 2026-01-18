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
def _flip(image, flip_index, scope_name):
    """Flip an image either horizontally or vertically.

  Outputs the contents of `image` flipped along the dimension `flip_index`.

  See also `reverse()`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    flip_index: 0 For vertical, 1 for horizontal.
    scope_name: string, scope name.

  Returns:
    A `Tensor` of the same type and shape as `image`.

  Raises:
    ValueError: if the shape of `image` not supported.
  """
    with ops.name_scope(None, scope_name, [image]):
        image = ops.convert_to_tensor(image, name='image')
        image = _AssertAtLeast3DImage(image)
        shape = image.get_shape()

        def f_rank3():
            return fix_image_flip_shape(image, array_ops.reverse(image, [flip_index]))

        def f_rank4():
            return array_ops.reverse(image, [flip_index + 1])
        if shape.ndims is None:
            rank = array_ops.rank(image)
            return tf_cond.cond(math_ops.equal(rank, 3), f_rank3, f_rank4)
        elif shape.ndims == 3:
            return f_rank3()
        elif shape.ndims == 4:
            return f_rank4()
        else:
            raise ValueError("'image' (shape %s)must have either 3 or 4 dimensions." % shape)