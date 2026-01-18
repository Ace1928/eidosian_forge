from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAccessLevelsResponse(_messages.Message):
    """A response to `ListAccessLevelsRequest`.

  Fields:
    accessLevels: List of the Access Level instances.
    nextPageToken: The pagination token to retrieve the next page of results.
      If the value is empty, no further results remain.
  """
    accessLevels = _messages.MessageField('AccessLevel', 1, repeated=True)
    nextPageToken = _messages.StringField(2)