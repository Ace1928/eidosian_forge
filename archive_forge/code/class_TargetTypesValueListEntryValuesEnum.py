from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetTypesValueListEntryValuesEnum(_messages.Enum):
    """TargetTypesValueListEntryValuesEnum enum type.

    Values:
      TARGET_TYPE_UNSPECIFIED: Do not use. You must set a target type
        explicitly.
      VIEWS: This entry applies to views in the dataset.
      ROUTINES: This entry applies to routines in the dataset.
    """
    TARGET_TYPE_UNSPECIFIED = 0
    VIEWS = 1
    ROUTINES = 2