from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DecryptRequest(_messages.Message):
    """Request message for KeyManagementService.Decrypt.

  Fields:
    additionalAuthenticatedData: Optional data that must match the data
      originally supplied in EncryptRequest.additional_authenticated_data.
    ciphertext: Required. The encrypted data originally returned in
      EncryptResponse.ciphertext.
  """
    additionalAuthenticatedData = _messages.BytesField(1)
    ciphertext = _messages.BytesField(2)