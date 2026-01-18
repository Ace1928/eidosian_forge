from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UserAgentValueValuesEnum(_messages.Enum):
    """The user agent used during scanning.

    Values:
      USER_AGENT_UNSPECIFIED: The user agent is unknown. Service will default
        to CHROME_LINUX.
      CHROME_LINUX: Chrome on Linux. This is the service default if
        unspecified.
      CHROME_ANDROID: Chrome on Android.
      SAFARI_IPHONE: Safari on IPhone.
    """
    USER_AGENT_UNSPECIFIED = 0
    CHROME_LINUX = 1
    CHROME_ANDROID = 2
    SAFARI_IPHONE = 3