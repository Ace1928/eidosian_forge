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
class ResourceVariable(BaseResourceVariable, composite_tensor.CompositeTensor):
    """Variable based on resource handles.

  See the [Variables How To](https://tensorflow.org/guide/variables)
  for a high level overview.

  A `ResourceVariable` allows you to maintain state across subsequent calls to
  session.run.

  The `ResourceVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  Just like any `Tensor`, variables created with
  `tf.Variable(use_resource=True)` can be used as inputs for other Ops in the
  graph. Additionally, all the operators overloaded for the `Tensor` class are
  carried over to variables, so you can also add nodes to the graph by just
  doing arithmetic on variables.

  Unlike ref-based variable, a ResourceVariable has well-defined semantics. Each
  usage of a ResourceVariable in a TensorFlow graph adds a read_value operation
  to the graph. The Tensors returned by a read_value operation are guaranteed to
  see all modifications to the value of the variable which happen in any
  operation on which the read_value depends on (either directly, indirectly, or
  via a control dependency) and guaranteed to not see any modification to the
  value of the variable from operations that depend on the read_value operation.
  Updates from operations that have no dependency relationship to the read_value
  operation might or might not be visible to read_value.

  For example, if there is more than one assignment to a ResourceVariable in
  a single session.run call there is a well-defined value for each operation
  which uses the variable's value if the assignments and the read are connected
  by edges in the graph. Consider the following example, in which two writes
  can cause tf.Variable and tf.ResourceVariable to behave differently:

  ```python
  a = tf.Variable(1.0, use_resource=True)
  a.initializer.run()

  assign = a.assign(2.0)
  with tf.control_dependencies([assign]):
    b = a.read_value()
  with tf.control_dependencies([b]):
    other_assign = a.assign(3.0)
  with tf.control_dependencies([other_assign]):
    # Will print 2.0 because the value was read before other_assign ran. If
    # `a` was a tf.Variable instead, 2.0 or 3.0 could be printed.
    tf.compat.v1.Print(b, [b]).eval()
  ```
  """

    def __init__(self, initial_value=None, trainable=None, collections=None, validate_shape=True, caching_device=None, name=None, dtype=None, variable_def=None, import_scope=None, constraint=None, distribute_strategy=None, synchronization=None, aggregation=None, shape=None, handle=None, experimental_enable_variable_lifting=None):
        """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. Can also be a callable with
        no argument that returns the initial value when called. (Note that
        initializer functions from init_ops.py must first be bound to a shape
        before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
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
      variable_def: `VariableDef` protocol buffer. If not None, recreates the
        `ResourceVariable` object with its contents. `variable_def` and other
        arguments (except for import_scope) are mutually exclusive.
      import_scope: Optional `string`. Name scope to add to the
        ResourceVariable. Only used when `variable_def` is provided.
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      distribute_strategy: The tf.distribute.Strategy this variable is being
        created inside of.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
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
      handle: (optional) The handle of a `tf.Variable`. If provided, only
        `trainable`, `shape`, `dtype`, and `handle` will be used to construct
        this `tf.Variable`.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, the default for the `collections` argument
    is `None`, which signifies that this `Variable` will not be added to any
    collections.
    @end_compatibility
    """
        if variable_def:
            if initial_value is not None:
                raise ValueError(f'The variable_def and initial_value args to `tf.Variable` are mutually exclusive, but got both: variable_def={variable_def},\ninitial_value={initial_value}')
            if context.executing_eagerly():
                raise ValueError(f'Creating a `tf.Variable` with a `variable_def` arg is not supported when eager execution is enabled. Got: variable_def={variable_def}')
            self._init_from_proto(variable_def, import_scope=import_scope, validate_shape=validate_shape)
        elif handle is not None:
            self._init_from_handle(trainable=trainable, shape=shape, dtype=dtype, handle=handle)
        else:
            self._init_from_args(initial_value=initial_value, trainable=trainable, collections=collections, caching_device=caching_device, name=name, dtype=dtype, constraint=constraint, synchronization=synchronization, aggregation=aggregation, shape=shape, distribute_strategy=distribute_strategy, validate_shape=validate_shape, experimental_enable_variable_lifting=experimental_enable_variable_lifting)

    @property
    def _type_spec(self):
        return VariableSpec.from_value(self)

    def _shape_invariant_to_type_spec(self, shape):
        return VariableSpec(shape, self.dtype, self.trainable)
    __composite_gradient__ = ResourceVariableGradient()

    def _init_from_args(self, initial_value=None, trainable=None, collections=None, caching_device=None, name=None, dtype=None, constraint=None, synchronization=None, aggregation=None, distribute_strategy=None, shape=None, validate_shape=True, experimental_enable_variable_lifting=None):
        """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound to
        a shape before being used here.)
      trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        Defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
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
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      distribute_strategy: DistributionStrategy under which this variable was
        created.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      experimental_enable_variable_lifting: Whether to lift the variable out if
        it's in a `tf.function`. Default is `True`. When this argument
        is `True`, variable creation will follow the behavior and
        restrictions described
        [here](https://www.tensorflow.org/guide/function#creating_tfvariables).
        If this argument is `False`, that description doesn't apply,
        and you can freely create and use the variable in the
        `tf.function`, as if it's a "mutable `tf.Tensor`". You can't
        return the variable though.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.

    @compatibility(eager)
    When Eager Execution is enabled, variables are never added to collections.
    It is not implicitly added to the `GLOBAL_VARIABLES` or
    `TRAINABLE_VARIABLES` collections, and the `collections` argument is
    ignored.
    @end_compatibility
    """
        synchronization, aggregation, trainable = variables.validate_synchronization_aggregation_trainable(synchronization, aggregation, trainable, name)
        if experimental_enable_variable_lifting is None:
            experimental_enable_variable_lifting = True
        if initial_value is None:
            raise ValueError('The `initial_value` arg to `tf.Variable` must be specified except when you are not providing a `variable_def`. You provided neither.')
        init_from_fn = callable(initial_value)
        if isinstance(initial_value, tensor_module.Tensor) and hasattr(initial_value, 'graph') and initial_value.graph.building_function:
            raise ValueError(f"Argument `initial_value` ({initial_value}) could not be lifted out of a `tf.function`. (Tried to create variable with name='{name}'). To avoid this error, when constructing `tf.Variable`s inside of `tf.function` you can create the `initial_value` tensor in a `tf.init_scope` or pass a callable `initial_value` (e.g., `tf.Variable(lambda : tf.truncated_normal([10, 40]))`). Please file a feature request if this restriction inconveniences you.")
        if collections is None:
            collections = [ops.GraphKeys.GLOBAL_VARIABLES]
        if not isinstance(collections, (list, tuple, set)):
            raise ValueError(f'collections argument to Variable constructor must be a list, tuple, or set. Got {collections} of type {type(collections)}')
        if constraint is not None and (not callable(constraint)):
            raise ValueError(f'Argument `constraint` must be None or a callable. a callable. Got a {type(constraint)}:  {constraint}')
        if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
            collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
        with ops.init_scope():
            self._in_graph_mode = not context.executing_eagerly()
        if experimental_enable_variable_lifting:
            maybe_init_scope = ops.init_scope
        else:
            maybe_init_scope = contextlib.nullcontext
        with maybe_init_scope():
            with ops.name_scope(name, 'Variable', [] if init_from_fn else [initial_value], skip_on_eager=False) as name:
                handle_name = ops.name_from_scope_name(name)
                if self._in_graph_mode:
                    shared_name = handle_name
                    unique_id = shared_name
                else:
                    unique_id = '%s_%d' % (handle_name, ops.uid())
                    shared_name = None
                device_context_manager = ops.device if self._in_graph_mode else ops.NullContextmanager
                attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(s=[compat.as_bytes('loc:@%s' % handle_name)]))
                with ops.get_default_graph()._attr_scope({'_class': attr}):
                    with ops.name_scope('Initializer'), device_context_manager(None):
                        if init_from_fn:
                            initial_value = initial_value()
                        if isinstance(initial_value, trackable.CheckpointInitialValue):
                            self._maybe_initialize_trackable()
                            self._update_uid = initial_value.checkpoint_position.restore_uid
                            initial_value = initial_value.wrapped_value
                        initial_value = ops.convert_to_tensor(initial_value, name='initial_value', dtype=dtype)
                    if shape is not None:
                        if not initial_value.shape.is_compatible_with(shape):
                            raise ValueError(f"In this `tf.Variable` creation, the initial value's shape ({initial_value.shape}) is not compatible with the explicitly supplied `shape` argument ({shape}).")
                    else:
                        shape = initial_value.shape
                    handle = eager_safe_variable_handle(initial_value=initial_value, shape=shape, shared_name=shared_name, name=name, graph_mode=self._in_graph_mode)
                    handle._parent_trackable = weakref.ref(self)
                    handle._name = handle_name + ':0'
                    handle._unique_id = unique_id
                if self._in_graph_mode and initial_value is not None and (initial_value.op._get_control_flow_context() is not None):
                    raise ValueError(f'The `initial_value` passed to `tf.Variable` {name} is from inside a control-flow  construct, such as a loop or conditional. When creating a `tf.Variable` inside a loop or conditional, use a lambda as the `initial_value`. Got: initial_value=({initial_value})')
                dtype = initial_value.dtype.base_dtype
                if self._in_graph_mode:
                    with ops.name_scope('IsInitialized'):
                        is_initialized_op = gen_resource_variable_ops.var_is_initialized_op(handle)
                    if initial_value is not None:
                        with ops.name_scope('Assign') as n, ops.colocate_with(None, ignore_existing=True), ops.device(handle.device):
                            initializer_op = gen_resource_variable_ops.assign_variable_op(handle, variables._try_guard_against_uninitialized_dependencies(name, initial_value), name=n)
                    with ops.name_scope('Read'):
                        with ops.device(handle.device):
                            value = gen_resource_variable_ops.read_variable_op(handle, dtype)
                            _maybe_set_handle_data(dtype, handle, value)
                        graph_element = value
                        if caching_device is not None:
                            with ops.colocate_with(None, ignore_existing=True):
                                with ops.device(caching_device):
                                    cached_value = array_ops.identity(value)
                        else:
                            cached_value = None
                else:
                    gen_resource_variable_ops.assign_variable_op(handle, initial_value)
                    is_initialized_op = None
                    initializer_op = None
                    graph_element = None
                    if caching_device:
                        with ops.device(caching_device):
                            cached_value = gen_resource_variable_ops.read_variable_op(handle, dtype)
                            _maybe_set_handle_data(dtype, handle, cached_value)
                    else:
                        cached_value = None
                if cached_value is not None:
                    cached_value._cached_variable = weakref.ref(self)
                if self._in_graph_mode:
                    ops.add_to_collections(collections, self)
                elif ops.GraphKeys.GLOBAL_STEP in collections:
                    ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, self)
            initial_value = initial_value if self._in_graph_mode else None
            super(ResourceVariable, self).__init__(trainable=trainable, shape=shape, dtype=dtype, handle=handle, synchronization=synchronization, constraint=constraint, aggregation=aggregation, distribute_strategy=distribute_strategy, name=name, unique_id=unique_id, handle_name=handle_name, graph_element=graph_element, initial_value=initial_value, initializer_op=initializer_op, is_initialized_op=is_initialized_op, cached_value=cached_value, caching_device=caching_device, validate_shape=validate_shape)

    def _init_from_proto(self, variable_def, import_scope=None, validate_shape=True):
        """Initializes from `VariableDef` proto."""
        assert not context.executing_eagerly()
        self._in_graph_mode = True
        assert isinstance(variable_def, variable_pb2.VariableDef)
        if not variable_def.is_resource:
            raise ValueError(f'The `variable_def` you passed to `tf.Variable` is Trying to restore a TF 1.x Reference Variable as a TF 2.x ResourceVariable. This is unsupported. Got variable_def={variable_def}')
        g = ops.get_default_graph()
        self._handle = g.as_graph_element(ops.prepend_name_scope(variable_def.variable_name, import_scope=import_scope), allow_operation=False)
        self._shape = tensor_shape.TensorShape(self._handle.op.get_attr('shape'))
        self._handle_name = self._handle.name
        self._unique_id = self._handle_name
        self._initializer_op = g.as_graph_element(ops.prepend_name_scope(variable_def.initializer_name, import_scope=import_scope))
        if hasattr(variable_def, 'initial_value_name') and variable_def.initial_value_name:
            self._initial_value = g.as_graph_element(ops.prepend_name_scope(variable_def.initial_value_name, import_scope=import_scope))
        else:
            self._initial_value = None
        synchronization, aggregation, trainable = variables.validate_synchronization_aggregation_trainable(variable_def.synchronization, variable_def.aggregation, variable_def.trainable, variable_def.variable_name)
        self._synchronization = synchronization
        self._aggregation = aggregation
        self._trainable = trainable
        if variable_def.snapshot_name:
            snapshot = g.as_graph_element(ops.prepend_name_scope(variable_def.snapshot_name, import_scope=import_scope))
            if snapshot.op.type != 'ReadVariableOp':
                self._cached_value = snapshot
            else:
                self._cached_value = None
            while snapshot.op.type != 'ReadVariableOp':
                snapshot = snapshot.op.inputs[0]
            self._graph_element = snapshot
        else:
            self._cached_value = None
            self._graph_element = g.get_tensor_by_name(self._handle.op.name + '/Read/ReadVariableOp:0')
        if variable_def.HasField('save_slice_info_def'):
            self._save_slice_info = variables.Variable.SaveSliceInfo(save_slice_info_def=variable_def.save_slice_info_def, import_scope=import_scope)
        else:
            self._save_slice_info = None
        self._caching_device = None
        self._dtype = dtypes.as_dtype(self._handle.op.get_attr('dtype'))
        self._constraint = None
        self._validate_shape = validate_shape

    def _init_from_handle(self, trainable=None, shape=None, dtype=None, handle=None):
        handle_data = get_eager_safe_handle_data(handle)
        if not handle_data.is_set:
            handle_data = handle_data_util.create_handle_data(shape, dtype)
            handle_data_util.set_handle_data(handle, handle_data)
        if hasattr(handle, '_name') and isinstance(handle._name, str):
            handle_name = handle._name.rstrip(':0')
        else:
            handle_name = None
        unique_id = getattr(handle, '_unique_id', None)
        super().__init__(trainable=trainable, shape=shape, dtype=dtype, handle=handle, unique_id=unique_id, handle_name=handle_name)