from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysGetRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysGetRequest object.

  Fields:
    name: The name of the CryptoKey to get.
  """
    name = _messages.StringField(1, required=True)