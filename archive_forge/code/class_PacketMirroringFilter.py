from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PacketMirroringFilter(_messages.Message):
    """A PacketMirroringFilter object.

  Enums:
    DirectionValueValuesEnum: Direction of traffic to mirror, either INGRESS,
      EGRESS, or BOTH. The default is BOTH.

  Fields:
    IPProtocols: Protocols that apply as filter on mirrored traffic. If no
      protocols are specified, all traffic that matches the specified CIDR
      ranges is mirrored. If neither cidrRanges nor IPProtocols is specified,
      all IPv4 traffic is mirrored.
    cidrRanges: One or more IPv4 or IPv6 CIDR ranges that apply as filter on
      the source (ingress) or destination (egress) IP in the IP header. If no
      ranges are specified, all IPv4 traffic that matches the specified
      IPProtocols is mirrored. If neither cidrRanges nor IPProtocols is
      specified, all IPv4 traffic is mirrored. To mirror all IPv4 and IPv6
      traffic, use "0.0.0.0/0,::/0".
    direction: Direction of traffic to mirror, either INGRESS, EGRESS, or
      BOTH. The default is BOTH.
  """

    class DirectionValueValuesEnum(_messages.Enum):
        """Direction of traffic to mirror, either INGRESS, EGRESS, or BOTH. The
    default is BOTH.

    Values:
      BOTH: Default, both directions are mirrored.
      EGRESS: Only egress traffic is mirrored.
      INGRESS: Only ingress traffic is mirrored.
    """
        BOTH = 0
        EGRESS = 1
        INGRESS = 2
    IPProtocols = _messages.StringField(1, repeated=True)
    cidrRanges = _messages.StringField(2, repeated=True)
    direction = _messages.EnumField('DirectionValueValuesEnum', 3)