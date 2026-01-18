from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyWeeklyCycleDayOfWeek(_messages.Message):
    """A ResourcePolicyWeeklyCycleDayOfWeek object.

  Enums:
    DayValueValuesEnum: Defines a schedule that runs on specific days of the
      week. Specify one or more days. The following options are available:
      MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY.

  Fields:
    day: Defines a schedule that runs on specific days of the week. Specify
      one or more days. The following options are available: MONDAY, TUESDAY,
      WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY.
    duration: [Output only] Duration of the time window, automatically chosen
      to be smallest possible in the given scenario.
    startTime: Time within the window to start the operations. It must be in
      format "HH:MM", where HH : [00-23] and MM : [00-00] GMT.
  """

    class DayValueValuesEnum(_messages.Enum):
        """Defines a schedule that runs on specific days of the week. Specify one
    or more days. The following options are available: MONDAY, TUESDAY,
    WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY.

    Values:
      FRIDAY: <no description>
      INVALID: <no description>
      MONDAY: <no description>
      SATURDAY: <no description>
      SUNDAY: <no description>
      THURSDAY: <no description>
      TUESDAY: <no description>
      WEDNESDAY: <no description>
    """
        FRIDAY = 0
        INVALID = 1
        MONDAY = 2
        SATURDAY = 3
        SUNDAY = 4
        THURSDAY = 5
        TUESDAY = 6
        WEDNESDAY = 7
    day = _messages.EnumField('DayValueValuesEnum', 1)
    duration = _messages.StringField(2)
    startTime = _messages.StringField(3)