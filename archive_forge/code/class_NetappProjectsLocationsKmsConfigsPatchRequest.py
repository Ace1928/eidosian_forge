from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsKmsConfigsPatchRequest(_messages.Message):
    """A NetappProjectsLocationsKmsConfigsPatchRequest object.

  Fields:
    kmsConfig: A KmsConfig resource to be passed as the request body.
    name: Identifier. Name of the KmsConfig.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the KmsConfig resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all fields will be overwritten.
  """
    kmsConfig = _messages.MessageField('KmsConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)