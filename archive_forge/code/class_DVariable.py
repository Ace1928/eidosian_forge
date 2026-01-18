import functools
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.DVariable', v1=[])
class DVariable(resource_variable_ops.ResourceVariable):
    """A replacement for tf.Variable which follows initial value placement.

    The class also handles restore/save operations in DTensor. Note that,
    DVariable may fall back to normal tf.Variable at this moment if
    `initial_value` is not a DTensor.
  """

    def __init__(self, initial_value, *args, dtype=None, **kwargs):
        """Overrides tf.Variable to fix VarHandleOp placements."""
        layout = kwargs.pop('layout', None)
        shape = kwargs.get('shape', None)
        if callable(initial_value):
            unwrapped = initial_value
            if issubclass(type(initial_value), functools.partial):
                unwrapped = initial_value.func
            if issubclass(type(unwrapped), trackable.CheckpointInitialValueCallable):
                if not shape or not layout:
                    raise ValueError('Expected shape and layout to be not None.')
                initial_value = api.call_with_layout(initial_value, layout, shard_info=trackable.ShardInfo(shape=shape, offset=[0] * len(shape)))
            else:
                initial_value = initial_value()
        if isinstance(initial_value, trackable.CheckpointInitialValue):
            initial_value = initial_value.wrapped_value
        initial_value = ops.convert_to_tensor(initial_value, dtype=dtype)
        variable_device = initial_value.device
        self._save_as_bf16 = False
        with ops.device(variable_device):
            if context.executing_eagerly():
                if api.is_dtensor(initial_value):
                    value_layout = api.fetch_layout(initial_value)
                    if layout is not None and layout != value_layout:
                        raise errors_impl.InvalidArgumentError(None, None, f'Conflicting layout are provided for initial value layout ({value_layout}) and variable ({layout}).')
                    layout = value_layout
                elif layout is not None:
                    initial_value = api.relayout(initial_value, layout)
                else:
                    raise errors_impl.InvalidArgumentError(None, None, 'Neither layout nor DTensor initial value are provided.')
                self.layout = layout
                with api.default_mesh(layout.mesh):
                    super(DVariable, self).__init__(initial_value, *args, dtype=dtype, **kwargs)
            else:
                if layout is not None:
                    initial_value = api.relayout(initial_value, layout)
                super(DVariable, self).__init__(initial_value, *args, dtype=dtype, **kwargs)

    @property
    def save_as_bf16(self):
        return self._save_as_bf16

    @save_as_bf16.setter
    def save_as_bf16(self, save_as_bf16):
        """Enables saving float32 as bfloat16."""
        self._save_as_bf16 = save_as_bf16 and self.dtype == dtypes.float32

    def _gather_saveables_for_checkpoint(self):
        return {trackable.VARIABLE_VALUE_KEY: functools.partial(_DVariableSaveable, self)}