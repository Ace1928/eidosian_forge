from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterStatusNatStatusNatRuleStatus(_messages.Message):
    """Status of a NAT Rule contained in this NAT.

  Fields:
    activeNatIps: A list of active IPs for NAT. Example: ["1.1.1.1",
      "179.12.26.133"].
    drainNatIps: A list of IPs for NAT that are in drain mode. Example:
      ["1.1.1.1", "179.12.26.133"].
    minExtraIpsNeeded: The number of extra IPs to allocate. This will be
      greater than 0 only if the existing IPs in this NAT Rule are NOT enough
      to allow all configured VMs to use NAT.
    numVmEndpointsWithNatMappings: Number of VM endpoints (i.e., NICs) that
      have NAT Mappings from this NAT Rule.
    ruleNumber: Rule number of the rule.
  """
    activeNatIps = _messages.StringField(1, repeated=True)
    drainNatIps = _messages.StringField(2, repeated=True)
    minExtraIpsNeeded = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    numVmEndpointsWithNatMappings = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    ruleNumber = _messages.IntegerField(5, variant=_messages.Variant.INT32)