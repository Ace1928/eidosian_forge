from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListEkmConnectionsResponse(_messages.Message):
    """Response message for EkmService.ListEkmConnections.

  Fields:
    ekmConnections: The list of EkmConnections.
    nextPageToken: A token to retrieve next page of results. Pass this value
      in ListEkmConnectionsRequest.page_token to retrieve the next page of
      results.
    totalSize: The total number of EkmConnections that matched the query.
  """
    ekmConnections = _messages.MessageField('EkmConnection', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)