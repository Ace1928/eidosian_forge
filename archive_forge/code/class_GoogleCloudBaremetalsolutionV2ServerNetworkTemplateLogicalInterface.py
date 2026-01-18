from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBaremetalsolutionV2ServerNetworkTemplateLogicalInterface(_messages.Message):
    """Logical interface.

  Enums:
    TypeValueValuesEnum: Interface type.

  Fields:
    name: Interface name. This is not a globally unique identifier. Name is
      unique only inside the ServerNetworkTemplate. This is of syntax or and
      forms part of the network template name.
    required: If true, interface must have network connected.
    type: Interface type.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Interface type.

    Values:
      INTERFACE_TYPE_UNSPECIFIED: Unspecified value.
      BOND: Bond interface type.
      NIC: NIC interface type.
    """
        INTERFACE_TYPE_UNSPECIFIED = 0
        BOND = 1
        NIC = 2
    name = _messages.StringField(1)
    required = _messages.BooleanField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)