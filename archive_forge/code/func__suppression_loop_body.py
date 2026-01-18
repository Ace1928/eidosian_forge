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
def _suppression_loop_body(boxes, iou_threshold, output_size, idx, tile_size):
    """Process boxes in the range [idx*tile_size, (idx+1)*tile_size).

  Args:
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    output_size: an int32 tensor of size [batch_size]. Representing the number
      of selected boxes for each batch.
    idx: an integer scalar representing induction variable.
    tile_size: an integer representing the number of boxes in a tile

  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
    with ops.name_scope('suppression_loop_body'):
        num_tiles = array_ops.shape(boxes)[1] // tile_size
        batch_size = array_ops.shape(boxes)[0]

        def cross_suppression_func(boxes, box_slice, iou_threshold, inner_idx):
            return _cross_suppression(boxes, box_slice, iou_threshold, inner_idx, tile_size)
        box_slice = array_ops.slice(boxes, [0, idx * tile_size, 0], [batch_size, tile_size, 4])
        _, box_slice, _, _ = while_loop.while_loop(lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx, cross_suppression_func, [boxes, box_slice, iou_threshold, constant_op.constant(0)])
        iou = _bbox_overlap(box_slice, box_slice)
        mask = array_ops.expand_dims(array_ops.reshape(math_ops.range(tile_size), [1, -1]) > array_ops.reshape(math_ops.range(tile_size), [-1, 1]), 0)
        iou *= math_ops.cast(math_ops.logical_and(mask, iou >= iou_threshold), iou.dtype)
        suppressed_iou, _, _, _ = while_loop.while_loop(lambda _iou, loop_condition, _iou_sum, _: loop_condition, _self_suppression, [iou, constant_op.constant(True), math_ops.reduce_sum(iou, [1, 2]), iou_threshold])
        suppressed_box = math_ops.reduce_sum(suppressed_iou, 1) > 0
        box_slice *= array_ops.expand_dims(1.0 - math_ops.cast(suppressed_box, box_slice.dtype), 2)
        mask = array_ops.reshape(math_ops.cast(math_ops.equal(math_ops.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
        boxes = array_ops.tile(array_ops.expand_dims(box_slice, [1]), [1, num_tiles, 1, 1]) * mask + array_ops.reshape(boxes, [batch_size, num_tiles, tile_size, 4]) * (1 - mask)
        boxes = array_ops.reshape(boxes, [batch_size, -1, 4])
        output_size += math_ops.reduce_sum(math_ops.cast(math_ops.reduce_any(box_slice > 0, [2]), dtypes.int32), [1])
    return (boxes, iou_threshold, output_size, idx + 1)