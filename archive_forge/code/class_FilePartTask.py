from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.command_lib.storage.tasks import task
class FilePartTask(task.Task):
    """Abstract class for handling a range of bytes in a file."""

    def __init__(self, source_resource, destination_resource, offset, length, component_number=None, total_components=None):
        """Initializes task.

    Args:
      source_resource (resource_reference.Resource): Source resource to copy.
      destination_resource (resource_reference.Resource): Target resource to
        copy to.
      offset (int): The index of the first byte in the range.
      length (int): The number of bytes in the range.
      component_number (int): If a multipart operation, indicates the
        component number.
      total_components (int): If a multipart operation, indicates the
        total number of components.
    """
        super(FilePartTask, self).__init__()
        self._source_resource = source_resource
        self._destination_resource = destination_resource
        self._offset = offset
        self._length = length
        self._component_number = component_number
        self._total_components = total_components

    @abc.abstractmethod
    def execute(self, task_status_queue=None):
        pass

    def __eq__(self, other):
        if not isinstance(other, FilePartTask):
            return NotImplemented
        return self._destination_resource == other._destination_resource and self._source_resource == other._source_resource and (self._offset == other._offset) and (self._length == other._length) and (self._component_number == other._component_number) and (self._total_components == other._total_components)