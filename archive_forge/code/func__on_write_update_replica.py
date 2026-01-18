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
def _on_write_update_replica(var, update_fn, value, **kwargs):
    """Updates variables with ON_WRITE synchronization in replica context."""
    if var.aggregation == vs.VariableAggregation.NONE:
        return update_fn(var._get_on_device_or_primary(), value, **kwargs)
    if not distribute_lib.get_strategy().extended._use_merge_call():
        if var.aggregation == vs.VariableAggregation.MEAN and (not var.dtype.is_floating) and tensor_util.is_tf_type(value):
            raise ValueError('Cannot update non-float variables with tf.VariableAggregation.MEAN aggregation in replica context. Either change the variable dtype to float or update it in cross-replica context.')
        aggregated_value = apply_aggregation_replica_context(value, var.aggregation, var)
        values_util.mark_as_unsaveable()
        return distribute_lib.get_replica_context()._update(var, update_fn, args=(aggregated_value,), kwargs=kwargs, group=True)
    else:

        def merge_fn(strategy, value, **kwargs):
            """Aggregate values and update all variables in cross replica context."""
            if var.aggregation == vs.VariableAggregation.MEAN and (not var.dtype.is_floating) and isinstance(value, PerReplica):
                raise ValueError('Cannot update non-float variables with tf.VariableAggregation.MEAN aggregation in replica context. Either change the variable dtype to float or update it in cross-replica context.')
            assert strategy == var.distribute_strategy
            v = values_util.apply_aggregation(strategy, value, var.aggregation, var)
            return var._update_cross_replica(update_fn, v, **kwargs)
        return distribute_lib.get_replica_context().merge_call(merge_fn, args=(value,), kwargs=kwargs)