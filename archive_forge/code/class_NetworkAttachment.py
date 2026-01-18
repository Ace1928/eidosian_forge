from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkAttachment(_messages.Message):
    """A VM attachment to a network.

  Enums:
    IpTypeValueValuesEnum: Output only. The type of this network attachment.
    IpVersionValueValuesEnum: Output only. The version of this IP address.

  Fields:
    ipAddress: Output only. The IP address on this network.
    ipType: Output only. The type of this network attachment.
    ipVersion: Output only. The version of this IP address.
    macAddress: Output only. The MAC address on this network.
    powerNetwork: Required. The name of the network attached to.
  """

    class IpTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of this network attachment.

    Values:
      IP_TYPE_UNSPECIFIED: The type of this ip is unknown.
      FIXED: The ip address is fixed.
      DYNAMIC: The ip address is dynamic.
    """
        IP_TYPE_UNSPECIFIED = 0
        FIXED = 1
        DYNAMIC = 2

    class IpVersionValueValuesEnum(_messages.Enum):
        """Output only. The version of this IP address.

    Values:
      IP_VERSION_UNSPECIFIED: The version of this ip is unknown.
      IPV4: The ip is an IPv4 address.
      IPV6: The ip is an IPv6 address.
    """
        IP_VERSION_UNSPECIFIED = 0
        IPV4 = 1
        IPV6 = 2
    ipAddress = _messages.StringField(1)
    ipType = _messages.EnumField('IpTypeValueValuesEnum', 2)
    ipVersion = _messages.EnumField('IpVersionValueValuesEnum', 3)
    macAddress = _messages.StringField(4)
    powerNetwork = _messages.StringField(5)