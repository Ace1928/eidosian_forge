from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export
class _RoundRobinStrategy:
    """Returns the next ps task index for placement in round-robin order.

  This class is not to be used directly by users.  See instead
  `replica_device_setter()` below.
  """

    def __init__(self, num_tasks):
        """Create a new `_RoundRobinStrategy`.

    Args:
      num_tasks: Number of ps tasks to cycle among.
    """
        self._num_tasks = num_tasks
        self._next_task = 0

    def __call__(self, unused_op):
        """Choose a ps task index for the given `Operation`.

    Args:
      unused_op: An `Operation` to be placed on ps.

    Returns:
      The next ps task index to use for the `Operation`. Returns the next
      index, in the range `[offset, offset + num_tasks)`.
    """
        task = self._next_task
        self._next_task = (self._next_task + 1) % self._num_tasks
        return task