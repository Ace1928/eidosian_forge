from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PaymentFundingModelValueValuesEnum(_messages.Enum):
    """DEPRECATED: This field is no longer supported and will eventually be
    removed when Apigee Hybrid 1.5/1.6 is no longer supported. Instead, use
    the `billingType` field inside `DeveloperMonetizationConfig` resource.
    Flag that specifies the billing account type, prepaid or postpaid.

    Values:
      PAYMENT_FUNDING_MODEL_UNSPECIFIED: Billing account type not specified.
      PREPAID: Prepaid billing account type. Developer pays in advance for the
        use of your API products. Funds are deducted from their prepaid
        account balance. **Note**: Not supported by Apigee at this time.
      POSTPAID: Postpaid billing account type. Developer is billed through an
        invoice after using your API products.
    """
    PAYMENT_FUNDING_MODEL_UNSPECIFIED = 0
    PREPAID = 1
    POSTPAID = 2