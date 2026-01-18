from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Gateway(_messages.Message):
    """Gateway represents a user facing component that serves as an entrance to
  enable connectivity.

  Enums:
    TypeValueValuesEnum: Required. The type of hosting used by the gateway.

  Fields:
    type: Required. The type of hosting used by the gateway.
    uri: Output only. Server-defined URI for this resource.
    userPort: Output only. User port reserved on the gateways for this
      connection, if not specified or zero, the default port is 19443.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of hosting used by the gateway.

    Values:
      TYPE_UNSPECIFIED: Default value. This value is unused.
      GCP_REGIONAL_MIG: Gateway hosted in a GCP regional managed instance
        group.
    """
        TYPE_UNSPECIFIED = 0
        GCP_REGIONAL_MIG = 1
    type = _messages.EnumField('TypeValueValuesEnum', 1)
    uri = _messages.StringField(2)
    userPort = _messages.IntegerField(3, variant=_messages.Variant.INT32)