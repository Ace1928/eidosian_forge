from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from six.moves import queue
def _get_flat_wildcard_results_iterator(self):
    """Iterates through items matching delete query, dividing into two lists.

    Separates objects and buckets, so we can return two separate iterators.

    Yields:
      True if resource found.
    """
    for name_expansion_result in self._name_expansion_iterator:
        resource = name_expansion_result.resource
        resource_url = resource.storage_url
        if resource_url.is_bucket():
            self._bucket_delete_tasks.put(delete_task.DeleteBucketTask(resource_url))
        elif isinstance(resource, resource_reference.ManagedFolderResource):
            self._managed_folder_delete_tasks.put(delete_task.DeleteManagedFolderTask(resource_url))
        else:
            self._object_delete_tasks.put(delete_task.DeleteObjectTask(resource_url, user_request_args=self._user_request_args))
        yield True