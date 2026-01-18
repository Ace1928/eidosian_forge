from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Connection(_messages.Message):
    """Contains information about the IP connection associated with the
  finding.

  Enums:
    ProtocolValueValuesEnum: IANA Internet Protocol Number such as TCP(6) and
      UDP(17).

  Fields:
    destinationIp: Destination IP address. Not present for sockets that are
      listening and not connected.
    destinationPort: Destination port. Not present for sockets that are
      listening and not connected.
    protocol: IANA Internet Protocol Number such as TCP(6) and UDP(17).
    sourceIp: Source IP address.
    sourcePort: Source port.
  """

    class ProtocolValueValuesEnum(_messages.Enum):
        """IANA Internet Protocol Number such as TCP(6) and UDP(17).

    Values:
      PROTOCOL_UNSPECIFIED: Unspecified protocol (not HOPOPT).
      ICMP: Internet Control Message Protocol.
      TCP: Transmission Control Protocol.
      UDP: User Datagram Protocol.
      GRE: Generic Routing Encapsulation.
      ESP: Encap Security Payload.
    """
        PROTOCOL_UNSPECIFIED = 0
        ICMP = 1
        TCP = 2
        UDP = 3
        GRE = 4
        ESP = 5
    destinationIp = _messages.StringField(1)
    destinationPort = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    protocol = _messages.EnumField('ProtocolValueValuesEnum', 3)
    sourceIp = _messages.StringField(4)
    sourcePort = _messages.IntegerField(5, variant=_messages.Variant.INT32)