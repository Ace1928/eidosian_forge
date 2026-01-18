from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsGatewaySecurityPoliciesPatchRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsGatewaySecurityPoliciesPatchRequest
  object.

  Fields:
    gatewaySecurityPolicy: A GatewaySecurityPolicy resource to be passed as
      the request body.
    name: Required. Name of the resource. Name is of the form projects/{projec
      t}/locations/{location}/gatewaySecurityPolicies/{gateway_security_policy
      } gateway_security_policy should match the
      pattern:(^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$).
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the GatewaySecurityPolicy resource by the update. The
      fields specified in the update_mask are relative to the resource, not
      the full request. A field will be overwritten if it is in the mask. If
      the user does not provide a mask then all fields will be overwritten.
  """
    gatewaySecurityPolicy = _messages.MessageField('GatewaySecurityPolicy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)