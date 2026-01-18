from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowedDeviceManagementLevelsValueListEntryValuesEnum(_messages.Enum):
    """AllowedDeviceManagementLevelsValueListEntryValuesEnum enum type.

    Values:
      MANAGEMENT_UNSPECIFIED: The device's management level is not specified
        or not known.
      NONE: The device is not managed.
      BASIC: Basic management is enabled, which is generally limited to
        monitoring and wiping the corporate account.
      COMPLETE: Complete device management. This includes more thorough
        monitoring and the ability to directly manage the device (such as
        remote wiping). This can be enabled through the Android Enterprise
        Platform.
    """
    MANAGEMENT_UNSPECIFIED = 0
    NONE = 1
    BASIC = 2
    COMPLETE = 3