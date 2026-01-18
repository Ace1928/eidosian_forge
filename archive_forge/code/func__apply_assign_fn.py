import abc
import enum
import functools
import itertools
import os
from tensorflow.core.framework import variable_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_should_use
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _apply_assign_fn(self, assign_fn, value):
    partition_axes = self._partition_axes()
    if len(partition_axes) > 1:
        raise NotImplementedError('Cannot do assign action along more than one dimension: %s.  Multi-axis partition assign action is not supported ' % str(partition_axes))
    if isinstance(value, list):
        assert len(value) == len(self._variable_list)
        value_list = value
    elif isinstance(value, PartitionedVariable):
        value_list = list(value)
    else:
        partition_ix = partition_axes[0]
        size_splits_list = [tensor_shape.dimension_value(var.shape[partition_ix]) for var in self._variable_list]
        value_list = array_ops.split(value, size_splits_list, axis=partition_ix)
    op_list = [assign_fn(var, value_list[idx]) for idx, var in enumerate(self._variable_list)]
    return op_list