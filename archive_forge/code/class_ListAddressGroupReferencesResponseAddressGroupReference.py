from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAddressGroupReferencesResponseAddressGroupReference(_messages.Message):
    """The Reference of AddressGroup.

  Fields:
    firewallPolicy: FirewallPolicy that is using the Address Group.
    rulePriority: Rule priority of the FirewallPolicy that is using the
      Address Group.
    ruleType: Type of the rule (applies only to FIREWALL_POLICY references)
    securityPolicy: Cloud Armor SecurityPolicy that is using the Address
      Group.
  """
    firewallPolicy = _messages.StringField(1)
    rulePriority = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    ruleType = _messages.StringField(3)
    securityPolicy = _messages.StringField(4)