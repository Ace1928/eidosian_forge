from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HeaderAction(_messages.Message):
    """Defines the addition and removal of HTTP headers for requests and
  responses.

  Fields:
    requestHeadersToAdd: Optional. A list of headers to add to the request
      prior to forwarding the request to the origin. You can add a maximum of
      25 request headers.
    requestHeadersToRemove: Optional. A list of header names to remove from
      the request before forwarding the request to the origin. You can specify
      up to 25 request headers to remove.
    responseHeadersToAdd: Optional. A list of headers to add to the response
      before sending it back to the client. You can add a maximum of 25
      response headers. Response headers are only sent to the client, and do
      not have an effect on the cache serving the response.
    responseHeadersToRemove: Optional. A list of headers to remove from the
      response before sending it back to the client. Response headers are only
      sent to the client, and do not have an effect on the cache serving the
      response. You can specify up to 25 response headers to remove.
  """
    requestHeadersToAdd = _messages.MessageField('HeaderActionAddHeader', 1, repeated=True)
    requestHeadersToRemove = _messages.MessageField('HeaderActionRemoveHeader', 2, repeated=True)
    responseHeadersToAdd = _messages.MessageField('HeaderActionAddHeader', 3, repeated=True)
    responseHeadersToRemove = _messages.MessageField('HeaderActionRemoveHeader', 4, repeated=True)