from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBillingBudgetsV1beta1ThresholdRule(_messages.Message):
    """ThresholdRule contains the definition of a threshold. Threshold rules
  define the triggering events used to generate a budget notification email.
  When a threshold is crossed (spend exceeds the specified percentages of the
  budget), budget alert emails are sent to the email recipients you specify in
  the [NotificationsRule](#notificationsrule). Threshold rules also affect the
  fields included in the [JSON data
  object](https://cloud.google.com/billing/docs/how-to/budgets-programmatic-
  notifications#notification_format) sent to a Pub/Sub topic. Threshold rules
  are _required_ if using email notifications. Threshold rules are _optional_
  if only setting a [`pubsubTopic` NotificationsRule](#NotificationsRule),
  unless you want your JSON data object to include data about the thresholds
  you set. For more information, see [set budget threshold rules and
  actions](https://cloud.google.com/billing/docs/how-to/budgets#budget-
  actions).

  Enums:
    SpendBasisValueValuesEnum: Optional. The type of basis used to determine
      if spend has passed the threshold. Behavior defaults to CURRENT_SPEND if
      not set.

  Fields:
    spendBasis: Optional. The type of basis used to determine if spend has
      passed the threshold. Behavior defaults to CURRENT_SPEND if not set.
    thresholdPercent: Required. Send an alert when this threshold is exceeded.
      This is a 1.0-based percentage, so 0.5 = 50%. Validation: non-negative
      number.
  """

    class SpendBasisValueValuesEnum(_messages.Enum):
        """Optional. The type of basis used to determine if spend has passed the
    threshold. Behavior defaults to CURRENT_SPEND if not set.

    Values:
      BASIS_UNSPECIFIED: Unspecified threshold basis.
      CURRENT_SPEND: Use current spend as the basis for comparison against the
        threshold.
      FORECASTED_SPEND: Use forecasted spend for the period as the basis for
        comparison against the threshold. FORECASTED_SPEND can only be set
        when the budget's time period is a Filter.calendar_period. It cannot
        be set in combination with Filter.custom_period.
    """
        BASIS_UNSPECIFIED = 0
        CURRENT_SPEND = 1
        FORECASTED_SPEND = 2
    spendBasis = _messages.EnumField('SpendBasisValueValuesEnum', 1)
    thresholdPercent = _messages.FloatField(2)