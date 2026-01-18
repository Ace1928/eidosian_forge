from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GenerateIdTokenResponse(_messages.Message):
    """A GenerateIdTokenResponse object.

  Fields:
    token: The OpenId Connect ID token.
  """
    token = _messages.StringField(1)