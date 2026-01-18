from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsTypeValueValuesEnum(_messages.Enum):
    """Required. The allowed OS type.

    Values:
      OS_UNSPECIFIED: The operating system of the device is not specified or
        not known.
      DESKTOP_MAC: A desktop Mac operating system.
      DESKTOP_WINDOWS: A desktop Windows operating system.
      DESKTOP_LINUX: A desktop Linux operating system.
      DESKTOP_CHROME_OS: A desktop ChromeOS operating system.
      ANDROID: An Android operating system.
      IOS: An iOS operating system.
    """
    OS_UNSPECIFIED = 0
    DESKTOP_MAC = 1
    DESKTOP_WINDOWS = 2
    DESKTOP_LINUX = 3
    DESKTOP_CHROME_OS = 4
    ANDROID = 5
    IOS = 6