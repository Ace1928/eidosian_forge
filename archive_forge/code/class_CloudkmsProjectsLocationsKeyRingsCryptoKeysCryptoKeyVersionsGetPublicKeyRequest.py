from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsGetPublicKeyRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsGetPublicK
  eyRequest object.

  Fields:
    name: Required. The name of the CryptoKeyVersion public key to get.
  """
    name = _messages.StringField(1, required=True)