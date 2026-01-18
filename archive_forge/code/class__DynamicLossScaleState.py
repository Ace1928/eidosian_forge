from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.mixed_precision import loss_scale as keras_loss_scale_module
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import base_delegate
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.util import nest
class _DynamicLossScaleState(trackable.Trackable):
    """The state of a dynamic loss scale."""

    def __init__(self, initial_loss_scale, growth_steps, multiplier):
        """Creates the dynamic loss scale."""
        super(_DynamicLossScaleState, self).__init__()
        self._initial_loss_scale = float(initial_loss_scale)
        self._growth_steps = int(growth_steps)
        self._multiplier = float(multiplier)
        self._weights = {}
        self._current_loss_scale = self._add_weight(name='current_loss_scale', dtype=dtypes.float32, initial_value=self._initial_loss_scale)
        self._counter = self._add_weight(name='good_steps', dtype=dtypes.int64, initial_value=0)

    def _add_weight(self, name, initial_value, dtype=None):
        """Adds a weight to this loss scale.

    Args:
      name: Variable name.
      initial_value: The variable's initial value.
      dtype: The type of the variable.

    Returns:
      A variable.

    Raises:
      RuntimeError: If a weight with `name` has already been added.
    """
        variable = variable_v1.VariableV1(initial_value=initial_value, name=name, dtype=dtype, trainable=False, use_resource=True, synchronization=variables.VariableSynchronization.AUTO, aggregation=variables.VariableAggregation.NONE)
        if context.executing_eagerly():
            graph_key = None
        else:
            graph = ops.get_default_graph()
            graph_key = graph._graph_key
        key = (name, graph_key)
        self._weights[key] = variable
        self._handle_deferred_dependencies(name=name, trackable=variable)
        backend.track_variable(variable)
        return variable

    def _trackable_children(self, save_type=trackable.SaveType.CHECKPOINT, **kwargs):
        """From Trackable. Gather graph-specific weights to save."""
        if context.executing_eagerly():
            graph_key = None
        else:
            graph = ops.get_default_graph()
            graph_key = graph._graph_key
        weights = {}
        for (name, g), v in sorted(self._weights.items(), key=lambda i: i[0][0]):
            if g == graph_key:
                weights[name] = v
        weights.update(super(_DynamicLossScaleState, self)._trackable_children(save_type, **kwargs))
        return weights

    def _lookup_dependency(self, name):
        """From Trackable. Find a weight in the current graph."""
        unconditional = super(_DynamicLossScaleState, self)._lookup_dependency(name)
        if unconditional is not None:
            return unconditional
        if context.executing_eagerly():
            graph_key = None
        else:
            graph = ops.get_default_graph()
            graph_key = graph._graph_key
        return self._weights.get((name, graph_key), None)

    @property
    def initial_loss_scale(self):
        return self._initial_loss_scale

    @property
    def growth_steps(self):
        return self._growth_steps

    @property
    def multiplier(self):
        return self._multiplier

    @property
    def current_loss_scale(self):
        """Returns the current loss scale as a float32 `tf.Variable`."""
        return self._current_loss_scale

    @property
    def counter(self):
        """Returns the counter as a float32 `tf.Variable`."""
        return self._counter

    def __call__(self):
        """Returns the current loss scale as a scalar `float32` tensor."""
        return tensor_conversion.convert_to_tensor_v2_with_dispatch(self._current_loss_scale)

    def update(self, grads):
        """Updates the value of the loss scale.

    Args:
      grads: A nested structure of unscaled gradients, each which is an
        all-reduced gradient of the loss with respect to a weight.

    Returns:
      update_op: In eager mode, None. In graph mode, an op to update the loss
        scale.
      should_apply_gradients: Either a bool or a scalar boolean tensor. If
        False, the caller should skip applying `grads` to the variables this
        step.
    """
        grads = nest.flatten(grads)
        if distribute_lib.has_strategy() and distribute_lib.in_cross_replica_context():
            distribution = distribute_lib.get_strategy()
            is_finite_per_replica = distribution.extended.call_for_each_replica(_is_all_finite, args=(grads,))
            is_finite = distribution.experimental_local_results(is_finite_per_replica)[0]
        else:
            is_finite = _is_all_finite(grads)

        def update_if_finite_grads():
            """Update assuming the gradients are finite."""

            def incr_loss_scale():
                new_loss_scale = self.current_loss_scale * self.multiplier
                return control_flow_ops.group(_assign_if_finite(self.current_loss_scale, new_loss_scale), self.counter.assign(0))
            return cond.cond(self.counter + 1 >= self.growth_steps, incr_loss_scale, lambda: _op_in_graph_mode(self.counter.assign_add(1)))

        def update_if_not_finite_grads():
            """Update assuming the gradients are nonfinite."""
            new_loss_scale = math_ops.maximum(self.current_loss_scale / self.multiplier, 1)
            return control_flow_ops.group(self.counter.assign(0), self.current_loss_scale.assign(new_loss_scale))
        update_op = cond.cond(is_finite, update_if_finite_grads, update_if_not_finite_grads)
        should_apply_gradients = is_finite
        return (update_op, should_apply_gradients)