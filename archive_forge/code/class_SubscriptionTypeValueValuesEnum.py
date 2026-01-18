from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubscriptionTypeValueValuesEnum(_messages.Enum):
    """Output only. DEPRECATED: This will eventually be replaced by
    BillingType. Subscription type of the Apigee organization. Valid values
    include trial (free, limited, and for evaluation purposes only) or paid
    (full subscription has been purchased). See [Apigee
    pricing](https://cloud.google.com/apigee/pricing/).

    Values:
      SUBSCRIPTION_TYPE_UNSPECIFIED: Subscription type not specified.
      PAID: Full subscription to Apigee has been purchased.
      TRIAL: Subscription to Apigee is free, limited, and used for evaluation
        purposes only.
    """
    SUBSCRIPTION_TYPE_UNSPECIFIED = 0
    PAID = 1
    TRIAL = 2