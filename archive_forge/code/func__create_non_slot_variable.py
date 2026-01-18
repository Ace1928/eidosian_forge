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
def _create_non_slot_variable(self, initial_value, name, colocate_with):
    """Add an extra variable, not associated with a slot."""
    eager = ops.executing_eagerly_outside_functions()
    graph = None if eager else colocate_with.graph
    key = (name, graph)
    v = self._non_slot_dict.get(key, None)
    if v is None:
        self._maybe_initialize_trackable()
        distribution_strategy = distribute_lib.get_strategy()
        with distribution_strategy.extended.colocate_vars_with(colocate_with):
            if eager:
                restored_initial_value = self._preload_simple_restoration(name=name)
                if restored_initial_value is not None:
                    initial_value = restored_initial_value
            v = variable_v1.VariableV1(initial_value, name=name, trainable=False, use_resource=resource_variable_ops.is_resource_variable(colocate_with))
        self._handle_deferred_dependencies(name=name, trackable=v)
        self._non_slot_dict[key] = v
    return v