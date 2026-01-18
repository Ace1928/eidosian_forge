from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesTasksCancelLeaseRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesTasksCancelLeaseRequest object.

  Fields:
    cancelLeaseRequest: A CancelLeaseRequest resource to be passed as the
      request body.
    name: Required. The task name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID
      `
  """
    cancelLeaseRequest = _messages.MessageField('CancelLeaseRequest', 1)
    name = _messages.StringField(2, required=True)