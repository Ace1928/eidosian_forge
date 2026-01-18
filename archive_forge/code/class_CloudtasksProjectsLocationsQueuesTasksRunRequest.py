from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesTasksRunRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesTasksRunRequest object.

  Fields:
    name: Required. The task name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID
      `
    runTaskRequest: A RunTaskRequest resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    runTaskRequest = _messages.MessageField('RunTaskRequest', 2)