from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRawEncryptRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRawEncrypt
  Request object.

  Fields:
    name: Required. The resource name of the CryptoKeyVersion to use for
      encryption.
    rawEncryptRequest: A RawEncryptRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    rawEncryptRequest = _messages.MessageField('RawEncryptRequest', 2)