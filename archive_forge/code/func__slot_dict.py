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
def _slot_dict(self, slot_name):
    """Returns a dict for caching slots created under the given name.

    Args:
      slot_name: Name for the slot.

    Returns:
      A dict that maps primary `Variable` objects to the slot created
      for that variable, under the given slot name.
    """
    named_slots = self._slots.get(slot_name, None)
    if named_slots is None:
        named_slots = {}
        self._slots[slot_name] = named_slots
    return named_slots