import copy
from typing import Optional
import weakref
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.types import trace
class OnReadPolicy(VariablePolicy):
    """Policy defined for `tf.VariableSynchronization.ON_READ` synchronization.

  This policy is created when `synchronization` is set to
  `tf.VariableSynchronization.ON_READ` and `aggregation` is set to any of the
  values allowed by the `tf.VariableAggregation` enum such as `NONE`, `SUM`,
  `MEAN` or `ONLY_FIRST_REPLICA`when creating a `tf.Variable` in `tf.distribute`
  scope.
  """

    def _is_mirrored(self):
        return False

    def value(self, var):
        with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
            if distribute_lib.in_cross_replica_context() and (not values_util.in_replica_update_context()):
                if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
                    return var._get_replica(0).value()
                return var._get_cross_replica()
            else:
                return var._get_on_device_or_primary().value()

    def _as_graph_element(self, var):
        with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
            if distribute_lib.in_cross_replica_context():
                return ops.convert_to_tensor(var._get_cross_replica())
        return var._get()._as_graph_element()

    def _get_cross_replica(self, var):
        if self._aggregation == vs.VariableAggregation.ONLY_FIRST_REPLICA:
            return var._get_replica(0)
        if self._aggregation == vs.VariableAggregation.SUM:
            values_util.mark_as_unsaveable()
        with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
            return var.distribute_strategy.reduce(reduce_util.ReduceOp.from_variable_aggregation(self._aggregation), var, axis=None)

    def _update_replica(self, var, update_fn, value, **kwargs):
        return update_fn(var._get_on_device_or_primary(), value, **kwargs)

    def _scatter_not_implemented(self, method):
        raise NotImplementedError(f"ON_READ variables doesn't support `{method}` in cross replica context")

    def assign_sub(self, var, value, use_locking=False, name=None, read_value=True):
        """Subtracts a value from this variable."""
        with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
            if distribute_lib.in_cross_replica_context() and (not values_util.in_replica_update_context()):
                values_util.mark_as_unsaveable()
                return values_util.on_read_assign_sub_cross_replica(var, value, read_value=read_value)
            else:
                return values_util.on_write_assign_sub(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, var, value, use_locking=False, name=None, read_value=True):
        """Adds a value to this variable."""
        with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
            if distribute_lib.in_cross_replica_context() and (not values_util.in_replica_update_context()):
                values_util.mark_as_unsaveable()
                return values_util.on_read_assign_add_cross_replica(var, value, read_value=read_value)
            else:
                return values_util.on_write_assign_add(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def assign(self, var, value, use_locking=False, name=None, read_value=True):
        with distribute_lib.enter_or_assert_strategy(var.distribute_strategy):
            if distribute_lib.in_cross_replica_context() and (not values_util.in_replica_update_context()):
                values_util.mark_as_unsaveable()
                return values_util.on_read_assign_cross_replica(var, value, read_value=read_value)
            else:
                return values_util.on_write_assign(var, value, use_locking=use_locking, name=name, read_value=read_value)

    def scatter_sub(self, *args, **kwargs):
        del args, kwargs
        self._scatter_not_implemented('scatter_sub')

    def scatter_add(self, *args, **kwargs):
        del args, kwargs
        self._scatter_not_implemented('scatter_add')

    def scatter_mul(self, *args, **kwargs):
        del args, kwargs
        self._scatter_not_implemented('scatter_mul')

    def scatter_div(self, *args, **kwargs):
        del args, kwargs
        self._scatter_not_implemented('scatter_div')

    def scatter_min(self, *args, **kwargs):
        del args, kwargs
        self._scatter_not_implemented('scatter_min')

    def scatter_max(self, *args, **kwargs):
        del args, kwargs
        self._scatter_not_implemented('scatter_max')

    def scatter_update(self, *args, **kwargs):
        del args, kwargs
        self._scatter_not_implemented('scatter_update')

    def get_saveable(self, var, primary_var, name):
        """Create a saveable object for the given variable."""
        return values_util.get_on_read_saveable(var, primary_var, name)

    def get_restore_ops(self, var, tensor):
        """Restore the same value into all variables."""
        return values_util.get_on_read_restore_ops(var, tensor, self._aggregation)