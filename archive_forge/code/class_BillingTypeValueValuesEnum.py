from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingTypeValueValuesEnum(_messages.Enum):
    """Billing type of the Apigee organization. See [Apigee
    pricing](https://cloud.google.com/apigee/pricing).

    Values:
      BILLING_TYPE_UNSPECIFIED: Billing type not specified.
      SUBSCRIPTION: A pre-paid subscription to Apigee.
      EVALUATION: Free and limited access to Apigee for evaluation purposes
        only.
      PAYG: Access to Apigee using a Pay-As-You-Go plan.
    """
    BILLING_TYPE_UNSPECIFIED = 0
    SUBSCRIPTION = 1
    EVALUATION = 2
    PAYG = 3