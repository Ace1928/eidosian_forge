import contextlib
import functools
import weakref
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.checkpoint import tensor_callable
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager import tape
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_module
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_resource_variable_ops import *
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
class UninitializedVariable(BaseResourceVariable):
    """A variable with no initializer."""

    def __init__(self, trainable=None, caching_device=None, name=None, shape=None, dtype=None, constraint=None, synchronization=None, aggregation=None, extra_handle_data=None, distribute_strategy=None, **unused_kwargs):
        """Creates the variable handle.

    Args:
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      shape: The variable's shape.
      dtype: The variable's dtype.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      extra_handle_data: Optional, another resource handle or Tensor with handle
        data to merge with `shape` and `dtype`.
      distribute_strategy: The tf.distribute.Strategy this variable is being
        created inside of.
    """
        with ops.init_scope():
            self._in_graph_mode = not context.executing_eagerly()
            with ops.name_scope(name, 'Variable', skip_on_eager=False) as name:
                handle_name = ops.name_from_scope_name(name)
                if self._in_graph_mode:
                    shared_name = handle_name
                    unique_id = shared_name
                else:
                    unique_id = '%s_%d' % (handle_name, ops.uid())
                    shared_name = None
                handle = _variable_handle_from_shape_and_dtype(shape=shape, dtype=dtype, shared_name=shared_name, name=name, graph_mode=self._in_graph_mode, initial_value=extra_handle_data)
                handle._parent_trackable = weakref.ref(self)
                handle._name = handle_name + ':0'
                handle._unique_id = unique_id
                if self._in_graph_mode:
                    with ops.name_scope('Read'):
                        with ops.device(handle.device):
                            value = gen_resource_variable_ops.read_variable_op(handle, dtype)
                            _maybe_set_handle_data(dtype, handle, value)
                        graph_element = value
                    ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, self)
                else:
                    graph_element = None
        super(UninitializedVariable, self).__init__(distribute_strategy=distribute_strategy, shape=shape, dtype=dtype, unique_id=unique_id, handle_name=handle_name, constraint=constraint, handle=handle, graph_element=graph_element, trainable=trainable, synchronization=synchronization, aggregation=aggregation, in_graph_mode=self._in_graph_mode, **unused_kwargs)