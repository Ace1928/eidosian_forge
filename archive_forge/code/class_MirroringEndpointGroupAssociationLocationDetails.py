from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MirroringEndpointGroupAssociationLocationDetails(_messages.Message):
    """Details about the association status in a specific location.

  Enums:
    StateValueValuesEnum: Output only. The association state in this location.

  Fields:
    location: Output only. The location.
    reason: Output only. The reason for an invalid state, if one is available.
    state: Output only. The association state in this location.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The association state in this location.

    Values:
      STATE_UNSPECIFIED: Not set.
      ACTIVE: Ready.
      FAILED: Failed to actuate the association.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        FAILED = 2
    location = _messages.StringField(1)
    reason = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)