from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProfileTypeValueValuesEnum(_messages.Enum):
    """Base profile type for text transformation.

    Values:
      PROFILE_TYPE_UNSPECIFIED: No profile provided. Same as BASIC.
      EMPTY: Empty profile which does not perform any transformations.
      BASIC: Automatically converts "DATE" infoTypes using a DateShiftConfig,
        and all other infoTypes using a ReplaceWithInfoTypeConfig.
    """
    PROFILE_TYPE_UNSPECIFIED = 0
    EMPTY = 1
    BASIC = 2