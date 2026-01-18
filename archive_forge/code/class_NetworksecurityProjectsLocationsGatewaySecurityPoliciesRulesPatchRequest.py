from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesPatchRequest(_messages.Message):
    """A
  NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesPatchRequest
  object.

  Fields:
    gatewaySecurityPolicyRule: A GatewaySecurityPolicyRule resource to be
      passed as the request body.
    name: Required. Immutable. Name of the resource. ame is the full resource
      name so projects/{project}/locations/{location}/gatewaySecurityPolicies/
      {gateway_security_policy}/rules/{rule} rule should match the pattern:
      (^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$).
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the GatewaySecurityPolicy resource by the update. The
      fields specified in the update_mask are relative to the resource, not
      the full request. A field will be overwritten if it is in the mask. If
      the user does not provide a mask then all fields will be overwritten.
  """
    gatewaySecurityPolicyRule = _messages.MessageField('GatewaySecurityPolicyRule', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)