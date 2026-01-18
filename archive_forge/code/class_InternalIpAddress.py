from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InternalIpAddress(_messages.Message):
    """A InternalIpAddress object.

  Enums:
    TypeValueValuesEnum: The type of the internal IP address.

  Fields:
    cidr: IP CIDR address or range.
    owner: The owner of the internal IP address.
    purpose: The purpose of the internal IP address if applicable.
    region: The region of the internal IP address if applicable.
    type: The type of the internal IP address.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the internal IP address.

    Values:
      PEER_RESERVED: Reserved IP ranges on peer networks.
      PEER_USED: Used IP ranges on peer networks, including peer subnetwork IP
        ranges.
      REMOTE_RESERVED: Reserved IP ranges on peer networks of peer networks.
      REMOTE_USED: Used IP ranges on peer networks of peer networks.
      RESERVED: Reserved IP ranges on local network.
      SUBNETWORK: Subnetwork IP ranges on local network.
      TYPE_UNSPECIFIED: <no description>
    """
        PEER_RESERVED = 0
        PEER_USED = 1
        REMOTE_RESERVED = 2
        REMOTE_USED = 3
        RESERVED = 4
        SUBNETWORK = 5
        TYPE_UNSPECIFIED = 6
    cidr = _messages.StringField(1)
    owner = _messages.StringField(2)
    purpose = _messages.StringField(3)
    region = _messages.StringField(4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)