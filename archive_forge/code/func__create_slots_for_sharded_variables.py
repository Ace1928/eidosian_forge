import abc
import contextlib
import functools
import warnings
from copy import deepcopy
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import initializers
from keras.src.engine import base_layer_utils
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.utils import generic_utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
def _create_slots_for_sharded_variables(self, var_list):
    """Add ShardedVariables to slots to later reconstruct for checkpointing.

        ShardedVariables don't have slot variables created for them; their
        shards do. This function allows users to call get_slot with a
        ShardedVariable input and receive a ShardedVariable output containing
        the appropriate slot vars.

        Iterate over the variables to find shards, and aggregate the sharded
        containers in a set. Add these ShardedVariables to _slots so that
        get_slot can retrieve the proper slot variables for their component
        shards, and reconstruct those into a ShardedVariable.

        Args:
          var_list: list or tuple of `Variable` objects that will be minimized
            using this optimizer.
        """
    sharded_vars = set()
    for var in var_list:
        if getattr(var, '_sharded_container', False):
            sharded_vars.add(var._sharded_container())
    for sharded_var in sharded_vars:
        sharded_key = _var_key(sharded_var)
        slot_dict = {}
        for slot in self.get_slot_names():
            slot_dict[slot] = sharded_var
        self._slots[sharded_key] = slot_dict