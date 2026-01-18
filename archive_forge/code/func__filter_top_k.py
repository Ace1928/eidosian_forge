from enum import Enum
import functools
import weakref
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as parallel_control_flow_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def _filter_top_k(x, k):
    """Filters top-k values in the last dim of x and set the rest to NEG_INF.

  Used for computing top-k prediction values in dense labels (which has the same
  shape as predictions) for recall and precision top-k metrics.

  Args:
    x: tensor with any dimensions.
    k: the number of values to keep.

  Returns:
    tensor with same shape and dtype as x.
  """
    _, top_k_idx = nn_ops.top_k(x, k, sorted=False)
    top_k_mask = math_ops.reduce_sum(array_ops.one_hot(top_k_idx, array_ops.shape(x)[-1], axis=-1), axis=-2)
    return x * top_k_mask + NEG_INF * (1 - top_k_mask)