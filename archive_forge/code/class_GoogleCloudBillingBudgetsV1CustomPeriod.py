from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBillingBudgetsV1CustomPeriod(_messages.Message):
    """All date times begin at 12 AM US and Canadian Pacific Time (UTC-8).

  Fields:
    endDate: Optional. The end date of the time period. Budgets with elapsed
      end date won't be processed. If unset, specifies to track all usage
      incurred since the start_date.
    startDate: Required. The start date must be after January 1, 2017.
  """
    endDate = _messages.MessageField('GoogleTypeDate', 1)
    startDate = _messages.MessageField('GoogleTypeDate', 2)