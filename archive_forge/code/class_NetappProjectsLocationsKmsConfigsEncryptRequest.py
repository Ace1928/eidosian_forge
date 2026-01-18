from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsKmsConfigsEncryptRequest(_messages.Message):
    """A NetappProjectsLocationsKmsConfigsEncryptRequest object.

  Fields:
    encryptVolumesRequest: A EncryptVolumesRequest resource to be passed as
      the request body.
    name: Required. Name of the KmsConfig.
  """
    encryptVolumesRequest = _messages.MessageField('EncryptVolumesRequest', 1)
    name = _messages.StringField(2, required=True)