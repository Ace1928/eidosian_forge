from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPartitionCursorsResponse(_messages.Message):
    """Response for ListPartitionCursors

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    partitionCursors: The partition cursors from this request.
  """
    nextPageToken = _messages.StringField(1)
    partitionCursors = _messages.MessageField('PartitionCursor', 2, repeated=True)