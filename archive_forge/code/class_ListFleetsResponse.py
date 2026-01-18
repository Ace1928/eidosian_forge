from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListFleetsResponse(_messages.Message):
    """Response message for the `GkeHub.ListFleetsResponse` method.

  Fields:
    fleets: The list of matching fleets.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages. The
      token is only valid for 1h.
  """
    fleets = _messages.MessageField('Fleet', 1, repeated=True)
    nextPageToken = _messages.StringField(2)