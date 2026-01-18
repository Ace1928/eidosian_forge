from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVersionsResponse(_messages.Message):
    """Response message for Versions.ListVersions.

  Fields:
    nextPageToken: Continuation token for fetching the next page of results.
    versions: The versions belonging to the requested service.
  """
    nextPageToken = _messages.StringField(1)
    versions = _messages.MessageField('Version', 2, repeated=True)