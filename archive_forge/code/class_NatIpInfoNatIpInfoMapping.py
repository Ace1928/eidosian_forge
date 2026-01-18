from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NatIpInfoNatIpInfoMapping(_messages.Message):
    """Contains information of a NAT IP.

  Enums:
    ModeValueValuesEnum: Specifies whether NAT IP is auto or manual.
    UsageValueValuesEnum: Specifies whether NAT IP is currently serving at
      least one endpoint or not.

  Fields:
    mode: Specifies whether NAT IP is auto or manual.
    natIp: NAT IP address. For example: 203.0.113.11.
    usage: Specifies whether NAT IP is currently serving at least one endpoint
      or not.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Specifies whether NAT IP is auto or manual.

    Values:
      AUTO: <no description>
      MANUAL: <no description>
    """
        AUTO = 0
        MANUAL = 1

    class UsageValueValuesEnum(_messages.Enum):
        """Specifies whether NAT IP is currently serving at least one endpoint or
    not.

    Values:
      IN_USE: <no description>
      UNUSED: <no description>
    """
        IN_USE = 0
        UNUSED = 1
    mode = _messages.EnumField('ModeValueValuesEnum', 1)
    natIp = _messages.StringField(2)
    usage = _messages.EnumField('UsageValueValuesEnum', 3)