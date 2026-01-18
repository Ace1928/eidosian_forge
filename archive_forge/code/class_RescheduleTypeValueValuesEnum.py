from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RescheduleTypeValueValuesEnum(_messages.Enum):
    """Required. If reschedule type is SPECIFIC_TIME, must set up
    schedule_time as well.

    Values:
      RESCHEDULE_TYPE_UNSPECIFIED: Not set.
      IMMEDIATE: If the user wants to schedule the maintenance to happen now.
      NEXT_AVAILABLE_WINDOW: If the user wants to use the existing maintenance
        policy to find the next available window.
      SPECIFIC_TIME: If the user wants to reschedule the maintenance to a
        specific time.
    """
    RESCHEDULE_TYPE_UNSPECIFIED = 0
    IMMEDIATE = 1
    NEXT_AVAILABLE_WINDOW = 2
    SPECIFIC_TIME = 3