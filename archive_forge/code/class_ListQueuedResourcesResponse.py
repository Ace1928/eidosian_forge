from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListQueuedResourcesResponse(_messages.Message):
    """Response for ListQueuedResources.

  Fields:
    nextPageToken: The next page token or empty if none.
    queuedResources: The listed queued resources.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    queuedResources = _messages.MessageField('QueuedResource', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)