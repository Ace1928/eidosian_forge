from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsConstraint(_messages.Message):
    """A restriction on the OS type and version of devices making requests.

  Enums:
    OsTypeValueValuesEnum: Required. The allowed OS type.

  Fields:
    minimumVersion: The minimum allowed OS version. If not set, any version of
      this OS satisfies the constraint. Format: `"major.minor.patch"`.
      Examples: `"10.5.301"`, `"9.2.1"`.
    osType: Required. The allowed OS type.
    requireVerifiedChromeOs: Only allows requests from devices with a verified
      Chrome OS. Verifications includes requirements that the device is
      enterprise-managed, conformant to domain policies, and the caller has
      permission to call the API targeted by the request.
  """

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
    minimumVersion = _messages.StringField(1)
    osType = _messages.EnumField('OsTypeValueValuesEnum', 2)
    requireVerifiedChromeOs = _messages.BooleanField(3)