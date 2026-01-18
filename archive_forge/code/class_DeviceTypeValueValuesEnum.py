from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeviceTypeValueValuesEnum(_messages.Enum):
    """Output only. Type of device.

    Values:
      DEVICE_TYPE_UNSPECIFIED: Unknown device type
      ANDROID: Device is an Android device
      IOS: Device is an iOS device
      GOOGLE_SYNC: Device is a Google Sync device.
      WINDOWS: Device is a Windows device.
      MAC_OS: Device is a MacOS device.
      LINUX: Device is a Linux device.
      CHROME_OS: Device is a ChromeOS device.
    """
    DEVICE_TYPE_UNSPECIFIED = 0
    ANDROID = 1
    IOS = 2
    GOOGLE_SYNC = 3
    WINDOWS = 4
    MAC_OS = 5
    LINUX = 6
    CHROME_OS = 7