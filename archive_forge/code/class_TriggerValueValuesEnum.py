from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TriggerValueValuesEnum(_messages.Enum):
    """Backfill job's triggering reason.

    Values:
      TRIGGER_UNSPECIFIED: Default value.
      AUTOMATIC: Object backfill job was triggered automatically according to
        the stream's backfill strategy.
      MANUAL: Object backfill job was triggered manually using the dedicated
        API.
    """
    TRIGGER_UNSPECIFIED = 0
    AUTOMATIC = 1
    MANUAL = 2