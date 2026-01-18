import functools
import numpy as np
from tensorflow.python.data.experimental.ops import random_access
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import random_seed
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _adjust_index(index, thresholds, offsets):
    """Adjusts index to account for elements to be skipped."""
    t_index = array_ops.shape(array_ops.boolean_mask(thresholds, math_ops.less_equal(thresholds, index)))[0] - 1
    return index + array_ops.gather(offsets, t_index)