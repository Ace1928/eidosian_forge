from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysDecryptRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysDecryptRequest object.

  Fields:
    decryptRequest: A DecryptRequest resource to be passed as the request
      body.
    name: Required. The resource name of the CryptoKey to use for decryption.
      The server will choose the appropriate version.
  """
    decryptRequest = _messages.MessageField('DecryptRequest', 1)
    name = _messages.StringField(2, required=True)