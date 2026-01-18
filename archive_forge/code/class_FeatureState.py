from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeatureState(_messages.Message):
    """FeatureState describes the high-level state of a Feature. It may be used
  to describe a Feature's state at the environ-level, or per-membershop,
  depending on the context.

  Enums:
    CodeValueValuesEnum: The high-level, machine-readable status of this
      Feature.

  Fields:
    code: The high-level, machine-readable status of this Feature.
    description: A human-readable description of the current status.
    updateTime: The time this status and any related Feature-specific details
      were updated.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """The high-level, machine-readable status of this Feature.

    Values:
      CODE_UNSPECIFIED: Unknown or not set.
      OK: The Feature is operating normally.
      WARNING: The Feature has encountered an issue, and is operating in a
        degraded state. The Feature may need intervention to return to normal
        operation. See the description and any associated Feature-specific
        details for more information.
      ERROR: The Feature is not operating or is in a severely degraded state.
        The Feature may need intervention to return to normal operation. See
        the description and any associated Feature-specific details for more
        information.
    """
        CODE_UNSPECIFIED = 0
        OK = 1
        WARNING = 2
        ERROR = 3
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    description = _messages.StringField(2)
    updateTime = _messages.StringField(3)