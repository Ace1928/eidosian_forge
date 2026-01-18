from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptRequest(_messages.Message):
    """Request message for KeyManagementService.Encrypt.

  Fields:
    additionalAuthenticatedData: Optional data that, if specified, must also
      be provided during decryption through
      DecryptRequest.additional_authenticated_data.  Must be no larger than
      64KiB.
    plaintext: Required. The data to encrypt. Must be no larger than 64KiB.
  """
    additionalAuthenticatedData = _messages.BytesField(1)
    plaintext = _messages.BytesField(2)