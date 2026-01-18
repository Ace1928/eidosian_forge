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
def _sort_scores_and_boxes(scores, boxes):
    """Sort boxes based their score from highest to lowest.

    Args:
      scores: a tensor with a shape of [batch_size, num_boxes] representing
        the scores of boxes.
      boxes: a tensor with a shape of [batch_size, num_boxes, 4] representing
        the boxes.
    Returns:
      sorted_scores: a tensor with a shape of [batch_size, num_boxes]
        representing the sorted scores.
      sorted_boxes: a tensor representing the sorted boxes.
      sorted_scores_indices: a tensor with a shape of [batch_size, num_boxes]
        representing the index of the scores in a sorted descending order.
    """
    with ops.name_scope('sort_scores_and_boxes'):
        sorted_scores_indices = sort_ops.argsort(scores, axis=1, direction='DESCENDING')
        sorted_scores = array_ops.gather(scores, sorted_scores_indices, axis=1, batch_dims=1)
        sorted_boxes = array_ops.gather(boxes, sorted_scores_indices, axis=1, batch_dims=1)
    return (sorted_scores, sorted_boxes, sorted_scores_indices)