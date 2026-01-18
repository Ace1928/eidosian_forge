from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysPatchRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysPatchRequest object.

  Fields:
    cryptoKey: A CryptoKey resource to be passed as the request body.
    name: Output only. The resource name for this CryptoKey in the format
      `projects/*/locations/*/keyRings/*/cryptoKeys/*`.
    updateMask: Required list of fields to be updated in this request.
  """
    cryptoKey = _messages.MessageField('CryptoKey', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)