from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PricingTypeValueValuesEnum(_messages.Enum):
    """How the cost is calculated.

    Values:
      PRICING_TYPE_UNSPECIFIED: Default pricing type.
      LIST_PRICE: The price listed by GCP for all customers.
      CUSTOM_PRICE: A price derived from past usage and billing.
    """
    PRICING_TYPE_UNSPECIFIED = 0
    LIST_PRICE = 1
    CUSTOM_PRICE = 2