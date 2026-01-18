from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksGetEffectiveFirewallsResponse(_messages.Message):
    """A NetworksGetEffectiveFirewallsResponse object.

  Fields:
    firewallPolicys: Effective firewalls from firewall policy.
    firewalls: Effective firewalls on the network.
    organizationFirewalls: Effective firewalls from organization policies.
  """
    firewallPolicys = _messages.MessageField('NetworksGetEffectiveFirewallsResponseEffectiveFirewallPolicy', 1, repeated=True)
    firewalls = _messages.MessageField('Firewall', 2, repeated=True)
    organizationFirewalls = _messages.MessageField('NetworksGetEffectiveFirewallsResponseOrganizationFirewallPolicy', 3, repeated=True)