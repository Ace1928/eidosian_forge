from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBillingBudgetsV1beta1BudgetAmount(_messages.Message):
    """The budgeted amount for each usage period.

  Fields:
    lastPeriodAmount: Use the last period's actual spend as the budget for the
      present period. LastPeriodAmount can only be set when the budget's time
      period is a Filter.calendar_period. It cannot be set in combination with
      Filter.custom_period.
    specifiedAmount: A specified amount to use as the budget. `currency_code`
      is optional. If specified when creating a budget, it must match the
      currency of the billing account. If specified when updating a budget, it
      must match the currency_code of the existing budget. The `currency_code`
      is provided on output.
  """
    lastPeriodAmount = _messages.MessageField('GoogleCloudBillingBudgetsV1beta1LastPeriodAmount', 1)
    specifiedAmount = _messages.MessageField('GoogleTypeMoney', 2)