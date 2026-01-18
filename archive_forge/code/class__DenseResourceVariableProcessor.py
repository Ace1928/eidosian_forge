import abc
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import slot_creator
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class _DenseResourceVariableProcessor(_OptimizableVariable):
    """Processor for dense ResourceVariables."""

    def __init__(self, v):
        self._v = v

    def target(self):
        return self._v

    def update_op(self, optimizer, g):
        if isinstance(g, indexed_slices.IndexedSlices):
            if self._v.constraint is not None:
                raise RuntimeError('Cannot use a constraint function on a sparse variable.')
            return optimizer._resource_apply_sparse_duplicate_indices(g.values, self._v, g.indices)
        update_op = optimizer._resource_apply_dense(g, self._v)
        if self._v.constraint is not None:
            with ops.control_dependencies([update_op]):
                return self._v.assign(self._v.constraint(self._v))
        else:
            return update_op