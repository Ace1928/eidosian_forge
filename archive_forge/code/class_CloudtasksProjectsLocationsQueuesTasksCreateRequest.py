from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesTasksCreateRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesTasksCreateRequest object.

  Fields:
    createTaskRequest: A CreateTaskRequest resource to be passed as the
      request body.
    parent: Required. The queue name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID` The queue
      must already exist.
  """
    createTaskRequest = _messages.MessageField('CreateTaskRequest', 1)
    parent = _messages.StringField(2, required=True)