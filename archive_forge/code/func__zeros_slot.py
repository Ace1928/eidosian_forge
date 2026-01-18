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
def _zeros_slot(self, var, slot_name, op_name):
    """Find or create a slot initialized with 0.0.

    Args:
      var: A `Variable` object.
      slot_name: Name for the slot.
      op_name: Name to use when scoping the Variable that
        needs to be created for the slot.

    Returns:
      A `Variable` object.
    """
    named_slots = self._slot_dict(slot_name)
    if _var_key(var) not in named_slots:
        new_slot_variable = slot_creator.create_zeros_slot(var, op_name, copy_xla_sharding=True)
        self._restore_slot_variable(slot_name=slot_name, variable=var, slot_variable=new_slot_variable)
        named_slots[_var_key(var)] = new_slot_variable
    return named_slots[_var_key(var)]