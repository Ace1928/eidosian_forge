import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
def _traverse_variables(self, to_traverse, visited):
    """Create the copied nodes and variables while traversing the nodes.

    This method performs a BFS to traverse the nodes while avoiding duplicated
    visits. Throughout the process, self._mapping, self._original_nodes, and
    self._var_pairs are populated.

    Args:
      to_traverse: A deque that stores the nodes to be traversed.
      visited: A list of nodes that have been visited.
    """
    while to_traverse:
        current_trackable = to_traverse.popleft()
        self._original_nodes.append(current_trackable)
        if isinstance(current_trackable, (Variable, ShardedVariable)):
            self._copy_trackable(current_trackable)
        if hasattr(current_trackable, _TPU_EMBEDDING_ATTR):
            self._handle_tpu_embedding(current_trackable)
        for child in current_trackable._trackable_children(save_type='checkpoint').values():
            if child in visited:
                continue
            visited.add(child)
            to_traverse.append(child)