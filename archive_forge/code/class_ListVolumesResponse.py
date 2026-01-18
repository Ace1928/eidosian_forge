from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVolumesResponse(_messages.Message):
    """Response message containing the list of storage volumes.

  Fields:
    nextPageToken: A token identifying a page of results from the server.
    unreachable: Locations that could not be reached.
    volumes: The list of storage volumes.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    volumes = _messages.MessageField('Volume', 3, repeated=True)