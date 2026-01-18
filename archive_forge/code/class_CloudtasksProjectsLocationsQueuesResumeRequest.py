from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesResumeRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesResumeRequest object.

  Fields:
    name: Required. The queue name. For example:
      `projects/PROJECT_ID/location/LOCATION_ID/queues/QUEUE_ID`
    resumeQueueRequest: A ResumeQueueRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    resumeQueueRequest = _messages.MessageField('ResumeQueueRequest', 2)