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
def _self_suppression(iou, _, iou_sum, iou_threshold):
    """Suppress boxes in the same tile.

     Compute boxes that cannot be suppressed by others (i.e.,
     can_suppress_others), and then use them to suppress boxes in the same tile.

  Args:
    iou: a tensor of shape [batch_size, num_boxes_with_padding] representing
    intersection over union.
    iou_sum: a scalar tensor.
    iou_threshold: a scalar tensor.

  Returns:
    iou_suppressed: a tensor of shape [batch_size, num_boxes_with_padding].
    iou_diff: a scalar tensor representing whether any box is supressed in
      this step.
    iou_sum_new: a scalar tensor of shape [batch_size] that represents
      the iou sum after suppression.
    iou_threshold: a scalar tensor.
  """
    batch_size = array_ops.shape(iou)[0]
    can_suppress_others = math_ops.cast(array_ops.reshape(math_ops.reduce_max(iou, 1) < iou_threshold, [batch_size, -1, 1]), iou.dtype)
    iou_after_suppression = array_ops.reshape(math_ops.cast(math_ops.reduce_max(can_suppress_others * iou, 1) < iou_threshold, iou.dtype), [batch_size, -1, 1]) * iou
    iou_sum_new = math_ops.reduce_sum(iou_after_suppression, [1, 2])
    return [iou_after_suppression, math_ops.reduce_any(iou_sum - iou_sum_new > iou_threshold), iou_sum_new, iou_threshold]