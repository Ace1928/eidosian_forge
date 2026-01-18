from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from six.moves import queue
def _resource_iterator(self, resource_queue):
    """Yields a resource from the queue."""
    resource_count = 0
    try:
        while not resource_queue.empty() or next(self._flat_wildcard_results_iterator):
            if not resource_queue.empty():
                resource_count += 1
                yield resource_queue.get()
    except StopIteration:
        pass
    if resource_count:
        progress_callbacks.workload_estimator_callback(self._task_status_queue, resource_count)