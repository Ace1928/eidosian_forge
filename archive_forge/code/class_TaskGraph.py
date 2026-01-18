from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
class TaskGraph:
    """Tracks dependencies between Task instances.

  See googlecloudsdk.command_lib.storage.tasks.task.Task for the definition of
  the Task class.

  The public methods in this class are thread safe.

  Attributes:
    is_empty (threading.Event): is_empty.is_set() is True when the graph has no
      tasks in it.
  """

    def __init__(self, top_level_task_limit):
        """Initializes a TaskGraph instance.

    Args:
      top_level_task_limit (int): A top-level task is a task that no other tasks
        depend on for completion (i.e. dependent_task_ids is None). Adding
        top-level tasks with TaskGraph.add will block until there are fewer than
        this number of top-level tasks in the graph.
    """
        self.is_empty = threading.Event()
        self.is_empty.set()
        self._lock = threading.RLock()
        self._task_wrappers_in_graph = {}
        self._top_level_task_semaphore = threading.Semaphore(top_level_task_limit)

    def add(self, task, dependent_task_ids=None):
        """Adds a task to the graph.

    Args:
      task (googlecloudsdk.command_lib.storage.tasks.task.Task): The task to be
        added.
      dependent_task_ids (Optional[List[Hashable]]): TaskWrapper.id attributes
        for tasks already in the graph that require the task being added to
        complete before being executed. This argument should be None for
        top-level tasks, which no other tasks depend on.

    Returns:
      A TaskWrapper instance for the task passed into this function, or None if
      task.parallel_processing_key was the same as another task's
      parallel_processing_key.

    Raises:
      InvalidDependencyError if any id in dependent_task_ids is not in the
      graph, or if a the add operation would have created a self-dependency.
    """
        is_top_level_task = dependent_task_ids is None
        if is_top_level_task:
            self._top_level_task_semaphore.acquire()
        with self._lock:
            if task.parallel_processing_key is not None:
                identifier = task.parallel_processing_key
            else:
                identifier = id(task)
            if identifier in self._task_wrappers_in_graph:
                if task.parallel_processing_key is not None:
                    log.status.Print('Skipping {} for {}. This can occur if a cp command results in multiple writes to the same resource.'.format(task.__class__.__name__, task.parallel_processing_key))
                else:
                    log.status.Print('Skipping {}. This is probably because due to a bug that caused it to be submitted for execution more than once.'.format(task.__class__.__name__))
                if is_top_level_task:
                    self._top_level_task_semaphore.release()
                return
            task_wrapper = TaskWrapper(identifier, task, dependent_task_ids)
            for task_id in dependent_task_ids or []:
                try:
                    self._task_wrappers_in_graph[task_id].dependency_count += 1
                except KeyError:
                    raise InvalidDependencyError
            self._task_wrappers_in_graph[task_wrapper.id] = task_wrapper
            self.is_empty.clear()
        return task_wrapper

    def complete(self, task_wrapper):
        """Recursively removes a task and its parents from the graph if possible.

    Tasks can be removed only if they have been submitted for execution and have
    no dependencies. Removing a task can affect dependent tasks in one of two
    ways, if the removal left the dependent tasks with no dependencies:
     - If the dependent task has already been submitted, it can also be removed.
     - If the dependent task has not already been submitted, it can be
       submitted for execution.

    This method removes all tasks that removing task_wrapper allows, and returns
    all tasks that can be submitted after removing task_wrapper.

    Args:
      task_wrapper (TaskWrapper): The task_wrapper instance to remove.

    Returns:
      An Iterable[TaskWrapper] that yields tasks that are submittable after
      completing task_wrapper.
    """
        with self._lock:
            if task_wrapper.dependency_count:
                return []
            if not task_wrapper.is_submitted:
                return [task_wrapper]
            del self._task_wrappers_in_graph[task_wrapper.id]
            if task_wrapper.dependent_task_ids is None:
                self._top_level_task_semaphore.release()
                if not self._task_wrappers_in_graph:
                    self.is_empty.set()
                return []
            submittable_tasks = []
            for task_id in task_wrapper.dependent_task_ids:
                dependent_task_wrapper = self._task_wrappers_in_graph[task_id]
                dependent_task_wrapper.dependency_count -= 1
                submittable_tasks += self.complete(dependent_task_wrapper)
            return submittable_tasks

    def update_from_executed_task(self, executed_task_wrapper, task_output):
        """Updates the graph based on the output of an executed task.

    If some googlecloudsdk.command_lib.storage.task.Task instance `a` returns
    the following iterables of tasks: [[b, c], [d, e]], we need to update the
    graph as follows to ensure they are executed appropriately.

           /-- d <-\\--/- b
      a <-/         \\/
          \\         /\\
           \\-- e <-/--\\- c

    After making these updates, `b` and `c` are ready for submission. If a task
    does not return any new tasks, then it will be removed from the graph,
    potentially freeing up tasks that depend on it for execution.

    See go/parallel-processing-in-gcloud-storage#heading=h.y4o7a9hcs89r for a
    more thorough description of the updates this method performs.

    Args:
      executed_task_wrapper (task_graph.TaskWrapper): Contains information about
        how a completed task fits into a dependency graph.
      task_output (Optional[task.Output]): Additional tasks and
        messages returned by the task in executed_task_wrapper.

    Returns:
      An Iterable[task_graph.TaskWrapper] containing tasks that are ready to be
      executed after performing graph updates.
    """
        with self._lock:
            if task_output is not None and task_output.messages is not None and (executed_task_wrapper.dependent_task_ids is not None):
                for task_id in executed_task_wrapper.dependent_task_ids:
                    dependent_task_wrapper = self._task_wrappers_in_graph[task_id]
                    dependent_task_wrapper.task.received_messages.extend(task_output.messages)
            if task_output is None or not task_output.additional_task_iterators:
                return self.complete(executed_task_wrapper)
            parent_tasks_for_next_layer = [executed_task_wrapper]
            for task_iterator in reversed(task_output.additional_task_iterators):
                dependent_task_ids = [task_wrapper.id for task_wrapper in parent_tasks_for_next_layer]
                parent_tasks_for_next_layer = [self.add(task, dependent_task_ids=dependent_task_ids) for task in task_iterator]
            return parent_tasks_for_next_layer