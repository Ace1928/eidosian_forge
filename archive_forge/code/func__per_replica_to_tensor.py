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
def _per_replica_to_tensor(var, dtype=None, name=None, as_ref=False):
    """Converts a `PerReplica` to a `Tensor`."""
    del name
    if dtype is not None and (not dtype.is_compatible_with(var.dtype)):
        raise ValueError('Incompatible type conversion requested to type {!r} for variable of type {!r}'.format(dtype.name, var.dtype.name))
    if as_ref:
        raise NotImplementedError("PerReplica doesn't support being used as a reference.")
    if distribute_lib.in_cross_replica_context() or not distribute_lib.has_strategy():
        raise ValueError('It looks like you are using a PerReplica object while not inside a replica context, which is not supported. Try running your op or function inside a replica context by using `strategy.run`')
    else:
        replica_id = values_util.get_current_replica_id_as_int()
        return var.values[replica_id]