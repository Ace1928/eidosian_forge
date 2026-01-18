from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Jwt(_messages.Message):
    """A Jwt object.

  Fields:
    compactJwt: The compact encoding of a JWS, which is always three base64
      encoded strings joined by periods. For details, see:
      https://tools.ietf.org/html/rfc7515.html#section-3.1
  """
    compactJwt = _messages.StringField(1)