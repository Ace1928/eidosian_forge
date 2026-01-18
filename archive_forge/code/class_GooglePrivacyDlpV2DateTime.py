from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DateTime(_messages.Message):
    """Message for a date time object. e.g. 2018-01-01, 5th August.

  Enums:
    DayOfWeekValueValuesEnum: Day of week

  Fields:
    date: One or more of the following must be set. Must be a valid date or
      time value.
    dayOfWeek: Day of week
    time: Time of day
    timeZone: Time zone
  """

    class DayOfWeekValueValuesEnum(_messages.Enum):
        """Day of week

    Values:
      DAY_OF_WEEK_UNSPECIFIED: The day of the week is unspecified.
      MONDAY: Monday
      TUESDAY: Tuesday
      WEDNESDAY: Wednesday
      THURSDAY: Thursday
      FRIDAY: Friday
      SATURDAY: Saturday
      SUNDAY: Sunday
    """
        DAY_OF_WEEK_UNSPECIFIED = 0
        MONDAY = 1
        TUESDAY = 2
        WEDNESDAY = 3
        THURSDAY = 4
        FRIDAY = 5
        SATURDAY = 6
        SUNDAY = 7
    date = _messages.MessageField('GoogleTypeDate', 1)
    dayOfWeek = _messages.EnumField('DayOfWeekValueValuesEnum', 2)
    time = _messages.MessageField('GoogleTypeTimeOfDay', 3)
    timeZone = _messages.MessageField('GooglePrivacyDlpV2TimeZone', 4)