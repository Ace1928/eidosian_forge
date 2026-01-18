from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsGetRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsGetRequest
  object.

  Fields:
    name: The name of the CryptoKeyVersion to get.
  """
    name = _messages.StringField(1, required=True)