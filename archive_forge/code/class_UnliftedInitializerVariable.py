import dataclasses
import functools
import os
import threading
import types as types_lib
import weakref
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import autograph_util
from tensorflow.python.eager.polymorphic_function import compiler_ir
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.tf_export import tf_export
class UnliftedInitializerVariable(resource_variable_ops.UninitializedVariable):
    """Variable which does not lift its initializer out of function context.

  Instances of this variable, when created, build a graph which runs their
  initializer inside a tf.cond(is_initialized) block.

  This can only be created during tracing compilation called from
  (eventually) eager mode. That is, non-function-building graphs are not
  supported.
  """

    def __init__(self, initial_value=None, trainable=None, caching_device=None, name=None, dtype=None, constraint=None, add_initializers_to=None, synchronization=None, aggregation=None, shape=None, **unused_kwargs):
        """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound to
        a shape before being used here.)
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      add_initializers_to: if not None and not in legacy graph mode, the
        initializer tensor will be added to this map in addition to adding the
        assignment to the function.
      synchronization: Indicates when a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
      RuntimeError: If called outside of a function definition.
    """
        with ops.init_scope():
            self._in_graph_mode = not context.executing_eagerly()
        if not ops.inside_function():
            resource_variable_ops.ResourceVariable.__init__(self, initial_value=initial_value, trainable=trainable, caching_device=caching_device, name=name, dtype=dtype, constraint=constraint)
            return
        if initial_value is None:
            raise ValueError('`initial_value` must be a Tensor or a Python object convertible to a Tensor. Got None.')
        init_from_fn = callable(initial_value)
        if constraint is not None and (not callable(constraint)):
            raise ValueError(f'`constraint` with type {type(constraint)} must be a callable.')
        with ops.name_scope(name, 'Variable', [] if init_from_fn else [initial_value]) as scope_name:
            with ops.name_scope('Initializer'):
                if init_from_fn:
                    initial_value = initial_value()
                if isinstance(initial_value, trackable.CheckpointInitialValue):
                    self._maybe_initialize_trackable()
                    self._update_uid = initial_value.checkpoint_position.restore_uid
                    initial_value = initial_value.wrapped_value
                initial_value = ops.convert_to_tensor(initial_value, name='initial_value', dtype=dtype)
            assert initial_value is not None
            if shape is None:
                shape = initial_value.shape
        super().__init__(trainable=trainable, caching_device=caching_device, name=name, shape=shape, dtype=initial_value.dtype, constraint=constraint, synchronization=synchronization, aggregation=aggregation, extra_handle_data=initial_value, **unused_kwargs)
        with ops.name_scope(scope_name):
            if self._in_graph_mode:
                with ops.init_scope():
                    outer_graph = ops.get_default_graph()
                func_graph = ops.get_default_graph()
                function_placeholders = func_graph.inputs + func_graph.internal_captures
                placeholder_ops = set([tensor.op for tensor in function_placeholders])
                lifted_initializer = lift_to_graph.lift_to_graph([initial_value], outer_graph, disallowed_placeholders=placeholder_ops)[initial_value]
                with ops.init_scope():
                    self._initial_value = lifted_initializer
                    with ops.name_scope('IsInitialized'):
                        self._is_initialized_op = resource_variable_ops.var_is_initialized_op(self._handle)
                    if initial_value is not None:
                        with ops.name_scope('Assign') as n, ops.colocate_with(self._handle):
                            self._initializer_op = resource_variable_ops.assign_variable_op(self._handle, lifted_initializer, name=n)
            elif context.executing_eagerly():
                with ops.name_scope('Assign') as n, ops.colocate_with(self._handle):
                    resource_variable_ops.assign_variable_op(self._handle, initial_value, name=n)
            else:
                if add_initializers_to is not None:
                    add_initializers_to.append((self, initial_value))

                def assign_fn():
                    with ops.name_scope('Assign') as n, ops.colocate_with(self._handle):
                        resource_variable_ops.assign_variable_op(self._handle, initial_value, name=n)
                    return ops.convert_to_tensor(1)

                def not_assign_fn():
                    return ops.convert_to_tensor(0)
                graph = ops.get_default_graph()
                graph.capture(self._handle, shape=())
                cond.cond(resource_variable_ops.var_is_initialized_op(self._handle), not_assign_fn, assign_fn)