import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _use_legacy_mode_for_tensor_equality(self):
    g = getattr(self, 'graph', None)
    return not (tensor.Tensor._USE_EQUALITY and ops.executing_eagerly_outside_functions() and (g is None or g.building_function))