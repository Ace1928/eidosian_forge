from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesPurgeRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesPurgeRequest object.

  Fields:
    name: Required. The queue name. For example:
      `projects/PROJECT_ID/location/LOCATION_ID/queues/QUEUE_ID`
    purgeQueueRequest: A PurgeQueueRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    purgeQueueRequest = _messages.MessageField('PurgeQueueRequest', 2)