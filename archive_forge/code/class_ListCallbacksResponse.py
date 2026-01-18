from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListCallbacksResponse(_messages.Message):
    """RPC response object for the ListCallbacks method.

  Fields:
    callbacks: The callbacks which match the request.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    callbacks = _messages.MessageField('Callback', 1, repeated=True)
    nextPageToken = _messages.StringField(2)