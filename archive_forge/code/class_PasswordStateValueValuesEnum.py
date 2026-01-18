from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PasswordStateValueValuesEnum(_messages.Enum):
    """Password state of the DeviceUser object

    Values:
      PASSWORD_STATE_UNSPECIFIED: Password state not set.
      PASSWORD_SET: Password set in object.
      PASSWORD_NOT_SET: Password not set in object.
    """
    PASSWORD_STATE_UNSPECIFIED = 0
    PASSWORD_SET = 1
    PASSWORD_NOT_SET = 2