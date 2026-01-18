from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1NatAddress(_messages.Message):
    """Apigee NAT(network address translation) address. A NAT address is a
  static external IP address used for Internet egress traffic.

  Enums:
    StateValueValuesEnum: Output only. State of the nat address.

  Fields:
    ipAddress: Output only. The static IPV4 address.
    name: Required. Resource ID of the NAT address.
    state: Output only. State of the nat address.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the nat address.

    Values:
      STATE_UNSPECIFIED: The resource is in an unspecified state.
      CREATING: The NAT address is being created.
      RESERVED: The NAT address is reserved but not yet used for Internet
        egress.
      ACTIVE: The NAT address is active and used for Internet egress.
      DELETING: The NAT address is being deleted.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        RESERVED = 2
        ACTIVE = 3
        DELETING = 4
    ipAddress = _messages.StringField(1)
    name = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)