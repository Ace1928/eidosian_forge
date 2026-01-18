from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage.tasks import task
class UploadTask(task.Task):
    """Base class for tasks that upload files."""

    def __init__(self, source_resource, destination_resource, length):
        """Initializes a task instance.

    Args:
      source_resource (resource_reference.FileObjectResource): The file to
        upload.
      destination_resource (resource_reference.ObjectResource|UnknownResource):
        Destination metadata for the upload.
      length (int): The size of source_resource.
    """
        super(UploadTask, self).__init__()
        self._source_resource = source_resource
        self._destination_resource = destination_resource
        self._length = length

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._source_resource == other._source_resource and self._destination_resource == other._destination_resource and (self._length == other._length)