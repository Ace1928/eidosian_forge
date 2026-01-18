from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRawDecryptRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRawDecrypt
  Request object.

  Fields:
    name: Required. The resource name of the CryptoKeyVersion to use for
      decryption.
    rawDecryptRequest: A RawDecryptRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    rawDecryptRequest = _messages.MessageField('RawDecryptRequest', 2)