from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeNetworkFirewallPoliciesGetRequest(_messages.Message):
    """A ComputeNetworkFirewallPoliciesGetRequest object.

  Fields:
    firewallPolicy: Name of the firewall policy to get.
    project: Project ID for this request.
  """
    firewallPolicy = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)