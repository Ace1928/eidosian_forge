from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BasicProfileValueValuesEnum(_messages.Enum):
    """Required. Profile which tells what the primitive action should be.

    Values:
      BASIC_PROFILE_UNSPECIFIED: If there is not a mentioned action for the
        target.
      ALLOW: Allow the matched traffic.
      DENY: Deny the matched traffic.
    """
    BASIC_PROFILE_UNSPECIFIED = 0
    ALLOW = 1
    DENY = 2