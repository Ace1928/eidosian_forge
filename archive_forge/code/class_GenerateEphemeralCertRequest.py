from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GenerateEphemeralCertRequest(_messages.Message):
    """Ephemeral certificate creation request.

  Fields:
    access_token: Optional. Access token to include in the signed certificate.
    public_key: PEM encoded public key to include in the signed certificate.
    readTime: Optional. Optional snapshot read timestamp to trade freshness
      for performance.
    validDuration: Optional. If set, it will contain the cert valid duration.
  """
    access_token = _messages.StringField(1)
    public_key = _messages.StringField(2)
    readTime = _messages.StringField(3)
    validDuration = _messages.StringField(4)