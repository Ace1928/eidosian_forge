import abc
import contextlib
import functools
import warnings
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def _create_or_restore_slot_variable(self, slot_variable_position, slot_name, variable):
    """Restore a slot variable's value, possibly creating it.

    Called when a variable which has an associated slot variable is created or
    restored. When executing eagerly, we create the slot variable with a
    restoring initializer.

    No new variables are created when graph building. Instead,
    _restore_slot_variable catches these after normal creation and adds restore
    ops to the graph. This method is nonetheless important when graph building
    for the case when a slot variable has already been created but `variable`
    has just been added to a dependency graph (causing us to realize that the
    slot variable needs to be restored).

    Args:
      slot_variable_position: A `trackable._CheckpointPosition` object
        indicating the slot variable `Trackable` object to be restored.
      slot_name: The name of this `Optimizer`'s slot to restore into.
      variable: The variable object this slot is being created for.
    """
    variable_key = _var_key(variable)
    slot_dict = self._slots.get(variable_key, {})
    slot_variable = slot_dict.get(slot_name, None)
    if slot_variable is None and context.executing_eagerly() and slot_variable_position.is_simple_variable() and (not ops.get_default_graph()._variable_creator_stack or self._distribution_strategy):
        initializer = trackable.CheckpointInitialValueCallable(checkpoint_position=slot_variable_position)
        slot_variable = self.add_slot(var=variable, initializer=initializer, slot_name=slot_name, shape=slot_variable_position.value_shape())
    if slot_variable is not None:
        slot_variable_position.restore(slot_variable)
    else:
        self._deferred_slot_restorations.setdefault(slot_name, {}).setdefault(variable_key, []).append(slot_variable_position)