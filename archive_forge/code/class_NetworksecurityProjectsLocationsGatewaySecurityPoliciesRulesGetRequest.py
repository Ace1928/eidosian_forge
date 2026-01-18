from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesGetRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesGetRequest
  object.

  Fields:
    name: Required. The name of the GatewaySecurityPolicyRule to retrieve.
      Format:
      projects/{project}/location/{location}/gatewaySecurityPolicies/*/rules/*
  """
    name = _messages.StringField(1, required=True)