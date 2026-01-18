from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Reschedule(_messages.Message):
    """A Reschedule object.

  Enums:
    RescheduleTypeValueValuesEnum: Required. The type of the reschedule.

  Fields:
    rescheduleType: Required. The type of the reschedule.
    scheduleTime: Optional. Timestamp when the maintenance shall be
      rescheduled to if reschedule_type=SPECIFIC_TIME, in [RFC
      3339](https://tools.ietf.org/html/rfc3339) format, for example
      `2012-11-15T16:19:00.094Z`.
  """

    class RescheduleTypeValueValuesEnum(_messages.Enum):
        """Required. The type of the reschedule.

    Values:
      RESCHEDULE_TYPE_UNSPECIFIED: <no description>
      IMMEDIATE: Reschedules maintenance to happen now (within 5 minutes).
      NEXT_AVAILABLE_WINDOW: Reschedules maintenance to occur within one week
        from the originally scheduled day and time.
      SPECIFIC_TIME: Reschedules maintenance to a specific time and day.
    """
        RESCHEDULE_TYPE_UNSPECIFIED = 0
        IMMEDIATE = 1
        NEXT_AVAILABLE_WINDOW = 2
        SPECIFIC_TIME = 3
    rescheduleType = _messages.EnumField('RescheduleTypeValueValuesEnum', 1)
    scheduleTime = _messages.StringField(2)