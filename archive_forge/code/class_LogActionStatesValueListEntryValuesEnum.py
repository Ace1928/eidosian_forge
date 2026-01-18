from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogActionStatesValueListEntryValuesEnum(_messages.Enum):
    """LogActionStatesValueListEntryValuesEnum enum type.

    Values:
      LOGGABLE_ACTION_STATE_UNSPECIFIED: Default value. This value is unused.
      SUCCEEDED: `LoggableAction` completed successfully. `SUCCEEDED` actions
        are logged as INFO.
      FAILED: `LoggableAction` terminated in an error state. `FAILED` actions
        are logged as ERROR.
    """
    LOGGABLE_ACTION_STATE_UNSPECIFIED = 0
    SUCCEEDED = 1
    FAILED = 2