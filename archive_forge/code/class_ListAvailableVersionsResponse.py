from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAvailableVersionsResponse(_messages.Message):
    """Response message for the list available versions request.

  Fields:
    availableVersions: Represents a list of versions that are supported.
    nextPageToken: Token to retrieve the next page of results or empty if
      there are no more results in the list.
  """
    availableVersions = _messages.MessageField('Version', 1, repeated=True)
    nextPageToken = _messages.StringField(2)