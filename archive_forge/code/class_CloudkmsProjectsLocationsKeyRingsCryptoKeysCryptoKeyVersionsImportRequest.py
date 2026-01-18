from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsImportRequest(_messages.Message):
    """A
  CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsImportRequest
  object.

  Fields:
    importCryptoKeyVersionRequest: A ImportCryptoKeyVersionRequest resource to
      be passed as the request body.
    parent: Required. The name of the CryptoKey to be imported into. The
      create permission is only required on this key when creating a new
      CryptoKeyVersion.
  """
    importCryptoKeyVersionRequest = _messages.MessageField('ImportCryptoKeyVersionRequest', 1)
    parent = _messages.StringField(2, required=True)