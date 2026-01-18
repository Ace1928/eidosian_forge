from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectiveQuotaGroup(_messages.Message):
    """An effective quota group contains both the metadata for a quota group as
  derived from the service config, and the effective limits in that group as
  calculated from producer and consumer overrides together with service
  defaults.

  Enums:
    BillingInteractionValueValuesEnum:

  Fields:
    baseGroup: The service configuration for this quota group, minus the quota
      limits, which are replaced by the effective limits below.
    billingInteraction: A BillingInteractionValueValuesEnum attribute.
    quotas: The usage and limit information for each limit within this quota
      group.
  """

    class BillingInteractionValueValuesEnum(_messages.Enum):
        """BillingInteractionValueValuesEnum enum type.

    Values:
      BILLING_INTERACTION_UNSPECIFIED: The interaction between this quota
        group and the project billing status is unspecified.
      NONBILLABLE_ONLY: This quota group is enforced only when the consumer
        project is not billable.
      BILLABLE_ONLY: This quota group is enforced only when the consumer
        project is billable.
      ANY_BILLING_STATUS: This quota group is enforced regardless of the
        consumer project's billing status.
    """
        BILLING_INTERACTION_UNSPECIFIED = 0
        NONBILLABLE_ONLY = 1
        BILLABLE_ONLY = 2
        ANY_BILLING_STATUS = 3
    baseGroup = _messages.MessageField('QuotaGroup', 1)
    billingInteraction = _messages.EnumField('BillingInteractionValueValuesEnum', 2)
    quotas = _messages.MessageField('QuotaInfo', 3, repeated=True)