from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SslCertsCreateEphemeralRequest(_messages.Message):
    """SslCerts create ephemeral certificate request.

  Fields:
    access_token: Access token to include in the signed certificate.
    public_key: PEM encoded public key to include in the signed certificate.
  """
    access_token = _messages.StringField(1)
    public_key = _messages.StringField(2)