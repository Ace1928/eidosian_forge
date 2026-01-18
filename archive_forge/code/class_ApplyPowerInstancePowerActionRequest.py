from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyPowerInstancePowerActionRequest(_messages.Message):
    """Message requesting to perform one of several power actions on an
  instance.

  Enums:
    ActionValueValuesEnum: Required. The action to perform on the instance.

  Fields:
    action: Required. The action to perform on the instance.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Required. The action to perform on the instance.

    Values:
      ACTION_UNSPECIFIED: No action was specified.
      START: Start the instance.
      STOP: Cleanly shut down the instance.
      SOFT_REBOOT: Cleanly reboot the instance.
      HARD_REBOOT: Hard reboot the instance.
      IMMEDIATE_SHUTDOWN: Immediately shut down the instance.
    """
        ACTION_UNSPECIFIED = 0
        START = 1
        STOP = 2
        SOFT_REBOOT = 3
        HARD_REBOOT = 4
        IMMEDIATE_SHUTDOWN = 5
    action = _messages.EnumField('ActionValueValuesEnum', 1)