from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsPatchRequest(_messages.Message):
    """A
  CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsPatchRequest
  object.

  Fields:
    cryptoKeyVersion: A CryptoKeyVersion resource to be passed as the request
      body.
    name: Output only. The resource name for this CryptoKeyVersion in the
      format
      `projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*`.
    updateMask: Required list of fields to be updated in this request.
  """
    cryptoKeyVersion = _messages.MessageField('CryptoKeyVersion', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)