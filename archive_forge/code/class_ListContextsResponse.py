from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListContextsResponse(_messages.Message):
    """Response for a ListContextsRequest.

  Fields:
    contexts: The list of contexts returned.
    nextPageToken: The page token used to query for the next page if one
      exists.
  """
    contexts = _messages.MessageField('Context', 1, repeated=True)
    nextPageToken = _messages.StringField(2)