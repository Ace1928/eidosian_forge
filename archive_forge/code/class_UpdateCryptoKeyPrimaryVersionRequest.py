from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateCryptoKeyPrimaryVersionRequest(_messages.Message):
    """Request message for KeyManagementService.UpdateCryptoKeyPrimaryVersion.

  Fields:
    cryptoKeyVersionId: The id of the child CryptoKeyVersion to use as
      primary.
  """
    cryptoKeyVersionId = _messages.StringField(1)