from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesListResponse(_messages.Message):
    """Database instances list response.

  Fields:
    items: List of database instance resources.
    kind: This is always `sql#instancesList`.
    nextPageToken: The continuation token, used to page through large result
      sets. Provide this value in a subsequent request to return the next page
      of results.
    warnings: List of warnings that occurred while handling the request.
  """
    items = _messages.MessageField('DatabaseInstance', 1, repeated=True)
    kind = _messages.StringField(2)
    nextPageToken = _messages.StringField(3)
    warnings = _messages.MessageField('ApiWarning', 4, repeated=True)