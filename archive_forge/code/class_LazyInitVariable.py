import threading
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
class LazyInitVariable(resource_variable_ops.BaseResourceVariable):
    """Lazily initialized variables.

    The major use case for this class is to serve as a memory efficient
    alternative for tf.Variable. The resource handle of this class is point to
    nothing, which mean it will raise error when its value is fetched in a eager
    context. Having said that, it will perform like a normal tf.Variable when
    using with graph tensor, like KerasTensor produced from tf.keras.Input.
    """

    def __init__(self, initial_value=None, trainable=None, collections=None, validate_shape=True, caching_device=None, name=None, dtype=None, variable_def=None, import_scope=None, constraint=None, distribute_strategy=None, synchronization=None, aggregation=None, shape=None, **kwargs):
        assert context.executing_eagerly()
        assert variable_def is None
        assert caching_device is None
        if initial_value is None:
            raise ValueError('The `initial_value` arg to `tf.Variable` must be specified except when you are not providing a `variable_def`. You provided neither.')
        if isinstance(initial_value, tensor.Tensor) and hasattr(initial_value, 'graph') and initial_value.graph.building_function:
            raise ValueError(f"Argument `initial_value` ({initial_value}) could not be lifted out of a `tf.function`. (Tried to create variable with name='{name}'). To avoid this error, when constructing `tf.Variable`s inside of `tf.function` you can create the `initial_value` tensor in a `tf.init_scope` or pass a callable `initial_value` (e.g., `tf.Variable(lambda : tf.truncated_normal([10, 40]))`). Please file a feature request if this restriction inconveniences you.")
        if constraint is not None and (not callable(constraint)):
            raise ValueError(f'Argument `constraint` must be None or a callable. a callable. Got a {type(constraint)}:  {constraint}')
        self._name = name
        initial_value, shape, dtype, handle, handle_name, unique_id = _infer_shape_dtype_and_create_handle(initial_value, shape, dtype, name)
        super().__init__(distribute_strategy=distribute_strategy, initial_value=initial_value, shape=shape, dtype=dtype, name=name, unique_id=unique_id, handle_name=handle_name, constraint=constraint, handle=handle, graph_element=None, trainable=trainable, synchronization=synchronization, aggregation=aggregation, in_graph_mode=False)

    def initialize(self):
        with ops.name_scope(self._name, 'Variable', skip_on_eager=False):
            with ops.colocate_with(self._handle), ops.name_scope('Initializer'):
                if callable(self._initial_value):
                    initial_value = self._initial_value()
                else:
                    initial_value = self._initial_value
                if not initial_value.shape.is_compatible_with(self._shape):
                    raise ValueError(f"In this `tf.Variable` creation, the initial value's shape ({initial_value.shape}) is not compatible with the explicitly supplied `shape` argument ({self._shape}).")
                assert self._dtype is initial_value.dtype.base_dtype
            gen_resource_variable_ops.assign_variable_op(self._handle, initial_value)

    def create_and_initialize(self):
        if callable(self._initial_value):
            initial_value = self._initial_value()
        with ops.device(initial_value.device):
            initial_value, shape, dtype, handle, handle_name, unique_id = _infer_shape_dtype_and_create_handle(initial_value, self._shape, self._dtype, self._name)
            self.initialize()
        super().__init__(trainable=self._trainable, shape=shape, dtype=dtype, handle=handle, synchronization=self._synchronization, constraint=self._constraint, aggregation=self._aggregation, distribute_strategy=self._distribute_strategy, name=self._name, unique_id=unique_id, handle_name=handle_name, graph_element=None, initial_value=initial_value, initializer_op=None, is_initialized_op=None, cached_value=None, caching_device=None)