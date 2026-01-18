from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmEndpointNatMappingsInterfaceNatMappings(_messages.Message):
    """Contain information of Nat mapping for an interface of this endpoint.

  Fields:
    drainNatIpPortRanges: List of all drain IP:port-range mappings assigned to
      this interface. These ranges are inclusive, that is, both the first and
      the last ports can be used for NAT. Example: ["2.2.2.2:12345-12355",
      "1.1.1.1:2234-2234"].
    natIpPortRanges: A list of all IP:port-range mappings assigned to this
      interface. These ranges are inclusive, that is, both the first and the
      last ports can be used for NAT. Example: ["2.2.2.2:12345-12355",
      "1.1.1.1:2234-2234"].
    numTotalDrainNatPorts: Total number of drain ports across all NAT IPs
      allocated to this interface. It equals to the aggregated port number in
      the field drain_nat_ip_port_ranges.
    numTotalNatPorts: Total number of ports across all NAT IPs allocated to
      this interface. It equals to the aggregated port number in the field
      nat_ip_port_ranges.
    ruleMappings: Information about mappings provided by rules in this NAT.
    sourceAliasIpRange: Alias IP range for this interface endpoint. It will be
      a private (RFC 1918) IP range. Examples: "10.33.4.55/32", or
      "192.168.5.0/24".
    sourceVirtualIp: Primary IP of the VM for this NIC.
  """
    drainNatIpPortRanges = _messages.StringField(1, repeated=True)
    natIpPortRanges = _messages.StringField(2, repeated=True)
    numTotalDrainNatPorts = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    numTotalNatPorts = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    ruleMappings = _messages.MessageField('VmEndpointNatMappingsInterfaceNatMappingsNatRuleMappings', 5, repeated=True)
    sourceAliasIpRange = _messages.StringField(6)
    sourceVirtualIp = _messages.StringField(7)