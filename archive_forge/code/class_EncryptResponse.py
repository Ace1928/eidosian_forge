from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptResponse(_messages.Message):
    """Response message for KeyManagementService.Encrypt.

  Fields:
    ciphertext: The encrypted data.
    name: The resource name of the CryptoKeyVersion used in encryption.
  """
    ciphertext = _messages.BytesField(1)
    name = _messages.StringField(2)