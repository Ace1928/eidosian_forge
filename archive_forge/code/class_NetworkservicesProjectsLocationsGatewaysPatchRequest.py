from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsGatewaysPatchRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsGatewaysPatchRequest object.

  Fields:
    gateway: A Gateway resource to be passed as the request body.
    name: Required. Name of the Gateway resource. It matches pattern
      `projects/*/locations/*/gateways/`.
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the Gateway resource by the update. The fields specified
      in the update_mask are relative to the resource, not the full request. A
      field will be overwritten if it is in the mask. If the user does not
      provide a mask then all fields will be overwritten.
  """
    gateway = _messages.MessageField('Gateway', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)