import contextlib
import copy
import functools
import threading
import weakref
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
def _maybe_build_distributed_table(self):
    """Create table objects and resources on each worker if hasn't been created."""
    with self._distributed_table_creation_lock:
        if not self._distributed_table:

            def create_copy():
                new_table = self._wrapped_creator()
                with self._has_resource_functions:
                    while not hasattr(self, '_restored_function') or any((method not in self._restored_function for method in TRACKABLE_RESOURCE_METHODS)):
                        self._has_resource_functions.wait()
                if hasattr(self, '_restored_function'):
                    with with_local_resource_restore_context(new_table):
                        for name, tf_function in self._restored_function.items():
                            setattr(new_table, name, tf_function)
                        init_op = new_table._initialize()
                        if not context.executing_eagerly():
                            ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
                ret = new_table.resource_handle
                return ret
            self._distributed_table = self._coordinator._create_per_worker_resources(create_copy)