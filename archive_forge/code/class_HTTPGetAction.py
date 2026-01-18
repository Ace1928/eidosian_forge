from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HTTPGetAction(_messages.Message):
    """HTTPGetAction describes an action based on HTTP Get requests.

  Fields:
    host: Not supported by Cloud Run.
    httpHeaders: Custom headers to set in the request. HTTP allows repeated
      headers.
    path: Path to access on the HTTP server.
    port: Port number to access on the container. Number must be in the range
      1 to 65535.
    scheme: Not supported by Cloud Run.
  """
    host = _messages.StringField(1)
    httpHeaders = _messages.MessageField('HTTPHeader', 2, repeated=True)
    path = _messages.StringField(3)
    port = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    scheme = _messages.StringField(5)