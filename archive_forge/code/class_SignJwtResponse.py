from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SignJwtResponse(_messages.Message):
    """A SignJwtResponse object.

  Fields:
    keyId: The ID of the key used to sign the JWT.
    signedJwt: The signed JWT.
  """
    keyId = _messages.StringField(1)
    signedJwt = _messages.StringField(2)