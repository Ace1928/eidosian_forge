from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeFirewallPoliciesGetRequest(_messages.Message):
    """A ComputeFirewallPoliciesGetRequest object.

  Fields:
    firewallPolicy: Name of the firewall policy to get.
  """
    firewallPolicy = _messages.StringField(1, required=True)