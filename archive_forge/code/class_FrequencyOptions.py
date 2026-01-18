from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FrequencyOptions(_messages.Message):
    """ReportConfig Resource: Options to setup frequency of report generation.

  Enums:
    FrequencyValueValuesEnum: Frequency of report generation.

  Fields:
    endDate: The date on which report generation should stop (Inclusive). UTC
      time zone.
    frequency: Frequency of report generation.
    startDate: The date from which report generation should start. UTC time
      zone.
  """

    class FrequencyValueValuesEnum(_messages.Enum):
        """Frequency of report generation.

    Values:
      FREQUENCY_UNSPECIFIED: Unspecified.
      DAILY: Report will be generated daily.
      WEEKLY: Report will be generated weekly.
    """
        FREQUENCY_UNSPECIFIED = 0
        DAILY = 1
        WEEKLY = 2
    endDate = _messages.MessageField('Date', 1)
    frequency = _messages.EnumField('FrequencyValueValuesEnum', 2)
    startDate = _messages.MessageField('Date', 3)