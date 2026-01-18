from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoutingConfig(_messages.Message):
    """RoutingConfig configures the behaviour of fleet logging feature.

  Enums:
    ModeValueValuesEnum: mode configures the logs routing mode.

  Fields:
    mode: mode configures the logs routing mode.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """mode configures the logs routing mode.

    Values:
      MODE_UNSPECIFIED: If UNSPECIFIED, fleet logging feature is disabled.
      COPY: logs will be copied to the destination project.
      MOVE: logs will be moved to the destination project.
    """
        MODE_UNSPECIFIED = 0
        COPY = 1
        MOVE = 2
    mode = _messages.EnumField('ModeValueValuesEnum', 1)