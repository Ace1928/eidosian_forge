from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteHttpDirectResponse(_messages.Message):
    """Static HTTP response object to be returned.

  Fields:
    bytesBody: Optional. Response body as bytes. Maximum body size is 4096B.
    status: Required. Status to return as part of HTTP Response. Must be a
      positive integer.
    stringBody: Optional. Response body as a string. Maximum body length is
      1024 characters.
  """
    bytesBody = _messages.BytesField(1)
    status = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    stringBody = _messages.StringField(3)