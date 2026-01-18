from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstancesGetEffectiveFirewallsResponse(_messages.Message):
    """A InstancesGetEffectiveFirewallsResponse object.

  Fields:
    firewallPolicys: Effective firewalls from firewall policies.
    firewalls: Effective firewalls on the instance.
    organizationFirewalls: Effective firewalls from organization policies.
  """
    firewallPolicys = _messages.MessageField('InstancesGetEffectiveFirewallsResponseEffectiveFirewallPolicy', 1, repeated=True)
    firewalls = _messages.MessageField('Firewall', 2, repeated=True)
    organizationFirewalls = _messages.MessageField('InstancesGetEffectiveFirewallsResponseOrganizationFirewallPolicy', 3, repeated=True)