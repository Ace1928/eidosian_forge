from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SignBlobResponse(_messages.Message):
    """A SignBlobResponse object.

  Fields:
    keyId: The ID of the key used to sign the blob.
    signedBlob: The signed blob.
  """
    keyId = _messages.StringField(1)
    signedBlob = _messages.BytesField(2)