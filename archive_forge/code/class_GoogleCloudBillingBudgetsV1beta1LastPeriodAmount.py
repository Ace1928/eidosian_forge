from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBillingBudgetsV1beta1LastPeriodAmount(_messages.Message):
    """Describes a budget amount targeted to the last Filter.calendar_period
  spend. At this time, the amount is automatically 100% of the last calendar
  period's spend; that is, there are no other options yet. Future
  configuration options will be described here (for example, configuring a
  percentage of last period's spend). LastPeriodAmount cannot be set for a
  budget configured with a Filter.custom_period.
  """