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
def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx, tile_size):
    """Suppress boxes between different tiles.

  Args:
    boxes: a tensor of shape [batch_size, num_boxes_with_padding, 4]
    box_slice: a tensor of shape [batch_size, tile_size, 4]
    iou_threshold: a scalar tensor
    inner_idx: a scalar tensor representing the tile index of the tile
      that is used to supress box_slice
    tile_size: an integer representing the number of boxes in a tile

  Returns:
    boxes: unchanged boxes as input
    box_slice_after_suppression: box_slice after suppression
    iou_threshold: unchanged
  """
    batch_size = array_ops.shape(boxes)[0]
    new_slice = array_ops.slice(boxes, [0, inner_idx * tile_size, 0], [batch_size, tile_size, 4])
    iou = _bbox_overlap(new_slice, box_slice)
    box_slice_after_suppression = array_ops.expand_dims(math_ops.cast(math_ops.reduce_all(iou < iou_threshold, [1]), box_slice.dtype), 2) * box_slice
    return (boxes, box_slice_after_suppression, iou_threshold, inner_idx + 1)