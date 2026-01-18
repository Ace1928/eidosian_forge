from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ActionTypeValueValuesEnum(_messages.Enum):
    """Allow or deny type.

    Values:
      ACTION_TYPE_UNSPECIFIED: Unspecified. Results in an error.
      ALLOW: Allowed action type.
      DENY: Deny action type.
    """
    ACTION_TYPE_UNSPECIFIED = 0
    ALLOW = 1
    DENY = 2