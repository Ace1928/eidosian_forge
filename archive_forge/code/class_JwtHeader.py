from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JwtHeader(_messages.Message):
    """[Deprecated] This message specifies a header location to extract JWT
  token. This message specifies a header location to extract JWT token.

  Fields:
    name: The HTTP header name.
    valuePrefix: The value prefix. The value format is "value_prefix" For
      example, for "Authorization: Bearer ", value_prefix="Bearer " with a
      space at the end.
  """
    name = _messages.StringField(1)
    valuePrefix = _messages.StringField(2)