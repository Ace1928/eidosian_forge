from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
class TaskWrapper:
    """Embeds a Task instance in a dependency graph.

  Attributes:
    id (Hashable): A unique identifier for this task wrapper.
    task (googlecloudsdk.command_lib.storage.tasks.task.Task): An instance of a
      task class.
    dependency_count (int): The number of unexecuted dependencies this task has,
      i.e. this node's in-degree in a graph where an edge from A to B indicates
      that A must be executed before B.
    dependent_task_ids (Optional[Iterable[Hashable]]): The id of the tasks that
      require this task to be completed for their own completion. This value
      should be None if no tasks depend on this one.
    is_submitted (bool): True if this task has been submitted for execution.
  """

    def __init__(self, task_id, task, dependent_task_ids):
        self.id = task_id
        self.task = task
        self.dependency_count = 0
        self.dependent_task_ids = dependent_task_ids
        self.is_submitted = False