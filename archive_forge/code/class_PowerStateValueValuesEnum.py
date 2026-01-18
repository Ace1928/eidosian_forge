from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PowerStateValueValuesEnum(_messages.Enum):
    """The power state of the VM at the moment list was taken.

    Values:
      POWER_STATE_UNSPECIFIED: Power state is not specified.
      ON: The VM is turned ON.
      OFF: The VM is turned OFF.
      SUSPENDED: The VM is suspended. This is similar to hibernation or sleep
        mode.
    """
    POWER_STATE_UNSPECIFIED = 0
    ON = 1
    OFF = 2
    SUSPENDED = 3