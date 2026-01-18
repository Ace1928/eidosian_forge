from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionNetworkFirewallPoliciesGetAssociationRequest(_messages.Message):
    """A ComputeRegionNetworkFirewallPoliciesGetAssociationRequest object.

  Fields:
    firewallPolicy: Name of the firewall policy to which the queried
      association belongs.
    name: The name of the association to get from the firewall policy.
    project: Project ID for this request.
    region: Name of the region scoping this request.
  """
    firewallPolicy = _messages.StringField(1, required=True)
    name = _messages.StringField(2)
    project = _messages.StringField(3, required=True)
    region = _messages.StringField(4, required=True)