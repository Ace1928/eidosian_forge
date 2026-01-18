from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsGatewaySecurityPoliciesDeleteRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsGatewaySecurityPoliciesDeleteRequest
  object.

  Fields:
    name: Required. A name of the GatewaySecurityPolicy to delete. Must be in
      the format
      `projects/{project}/locations/{location}/gatewaySecurityPolicies/*`.
  """
    name = _messages.StringField(1, required=True)