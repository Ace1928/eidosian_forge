from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOtherCloudConnectionsResponse(_messages.Message):
    """Response to list other-cloud connections.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    otherCloudConnections: The other-cloud connections from the specified
      parent or all parents.
  """
    nextPageToken = _messages.StringField(1)
    otherCloudConnections = _messages.MessageField('OtherCloudConnection', 2, repeated=True)