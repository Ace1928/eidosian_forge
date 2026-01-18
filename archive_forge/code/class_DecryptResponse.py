from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DecryptResponse(_messages.Message):
    """Response message for KeyManagementService.Decrypt.

  Fields:
    plaintext: The decrypted data originally supplied in
      EncryptRequest.plaintext.
  """
    plaintext = _messages.BytesField(1)