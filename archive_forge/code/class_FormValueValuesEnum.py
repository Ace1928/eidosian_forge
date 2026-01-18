from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FormValueValuesEnum(_messages.Enum):
    """Whether this block is used by physical or virtual devices

    Values:
      DEVICE_FORM_UNSPECIFIED: Do not use. For proto versioning only.
      VIRTUAL: Android virtual device using Compute Engine native
        virtualization. Firebase Test Lab only.
      PHYSICAL: Actual hardware.
      EMULATOR: Android virtual device using emulator in nested
        virtualization. Equivalent to Android Studio.
    """
    DEVICE_FORM_UNSPECIFIED = 0
    VIRTUAL = 1
    PHYSICAL = 2
    EMULATOR = 3