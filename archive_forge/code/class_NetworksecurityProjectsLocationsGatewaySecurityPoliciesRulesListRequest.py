from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesListRequest(_messages.Message):
    """A
  NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesListRequest
  object.

  Fields:
    pageSize: Maximum number of GatewaySecurityPolicyRules to return per call.
    pageToken: The value returned by the last
      'ListGatewaySecurityPolicyRulesResponse' Indicates that this is a
      continuation of a prior 'ListGatewaySecurityPolicyRules' call, and that
      the system should return the next page of data.
    parent: Required. The project, location and GatewaySecurityPolicy from
      which the GatewaySecurityPolicyRules should be listed, specified in the
      format `projects/{project}/locations/{location}/gatewaySecurityPolicies/
      {gatewaySecurityPolicy}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)