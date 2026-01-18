from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3ScheduledRun(_messages.Message):
    """Configuration for scheduled runs.

  Enums:
    ScheduledRunOptionValueValuesEnum: Required. The scheduled run option of
      the crawler.

  Fields:
    scheduledRunOption: Required. The scheduled run option of the crawler.
  """

    class ScheduledRunOptionValueValuesEnum(_messages.Enum):
        """Required. The scheduled run option of the crawler.

    Values:
      SCHEDULED_RUN_OPTION_UNSPECIFIED: Unspecified scheduled run option.
      DAILY: Daily scheduled run option.
      WEEKLY: Weekly scheduled run option.
    """
        SCHEDULED_RUN_OPTION_UNSPECIFIED = 0
        DAILY = 1
        WEEKLY = 2
    scheduledRunOption = _messages.EnumField('ScheduledRunOptionValueValuesEnum', 1)