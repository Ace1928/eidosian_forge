from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesTasksBufferRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesTasksBufferRequest object.

  Fields:
    bufferTaskRequest: A BufferTaskRequest resource to be passed as the
      request body.
    queue: Required. The parent queue name. For example:
      projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID` The queue
      must already exist.
    taskId: Optional. Task ID for the task being created. If not provided,
      Cloud Tasks generates an ID for the task.
  """
    bufferTaskRequest = _messages.MessageField('BufferTaskRequest', 1)
    queue = _messages.StringField(2, required=True)
    taskId = _messages.StringField(3, required=True)