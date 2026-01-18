from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterWeeklyMaintenanceWindow(_messages.Message):
    """Time window specified for weekly operations.

  Enums:
    DayValueValuesEnum: Allows to define schedule that runs specified day of
      the week.

  Fields:
    day: Allows to define schedule that runs specified day of the week.
    duration: Duration of the time window.
    startTime: Start time of the window in UTC.
  """

    class DayValueValuesEnum(_messages.Enum):
        """Allows to define schedule that runs specified day of the week.

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
    day = _messages.EnumField('DayValueValuesEnum', 1)
    duration = _messages.StringField(2)
    startTime = _messages.MessageField('TimeOfDay', 3)