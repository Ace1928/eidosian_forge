from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRestoreRequest(_messages.Message):
    """A
  CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRestoreRequest
  object.

  Fields:
    name: The resource name of the CryptoKeyVersion to restore.
    restoreCryptoKeyVersionRequest: A RestoreCryptoKeyVersionRequest resource
      to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    restoreCryptoKeyVersionRequest = _messages.MessageField('RestoreCryptoKeyVersionRequest', 2)