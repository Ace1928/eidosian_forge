from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmEndpointNatMappingsInterfaceNatMappingsNatRuleMappings(_messages.Message):
    """Contains information of NAT Mappings provided by a NAT Rule.

  Fields:
    drainNatIpPortRanges: List of all drain IP:port-range mappings assigned to
      this interface by this rule. These ranges are inclusive, that is, both
      the first and the last ports can be used for NAT. Example:
      ["2.2.2.2:12345-12355", "1.1.1.1:2234-2234"].
    natIpPortRanges: A list of all IP:port-range mappings assigned to this
      interface by this rule. These ranges are inclusive, that is, both the
      first and the last ports can be used for NAT. Example:
      ["2.2.2.2:12345-12355", "1.1.1.1:2234-2234"].
    numTotalDrainNatPorts: Total number of drain ports across all NAT IPs
      allocated to this interface by this rule. It equals the aggregated port
      number in the field drain_nat_ip_port_ranges.
    numTotalNatPorts: Total number of ports across all NAT IPs allocated to
      this interface by this rule. It equals the aggregated port number in the
      field nat_ip_port_ranges.
    ruleNumber: Rule number of the NAT Rule.
  """
    drainNatIpPortRanges = _messages.StringField(1, repeated=True)
    natIpPortRanges = _messages.StringField(2, repeated=True)
    numTotalDrainNatPorts = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    numTotalNatPorts = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    ruleNumber = _messages.IntegerField(5, variant=_messages.Variant.INT32)