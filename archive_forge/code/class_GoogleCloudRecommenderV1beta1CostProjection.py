from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1beta1CostProjection(_messages.Message):
    """Contains metadata about how much money a recommendation can save or
  incur.

  Enums:
    PricingTypeValueValuesEnum: How the cost is calculated.

  Fields:
    cost: An approximate projection on amount saved or amount incurred.
      Negative cost units indicate cost savings and positive cost units
      indicate increase. See google.type.Money documentation for
      positive/negative units. A user's permissions may affect whether the
      cost is computed using list prices or custom contract prices.
    costInLocalCurrency: The approximate cost savings in the billing account's
      local currency.
    duration: Duration for which this cost applies.
    pricingType: How the cost is calculated.
  """

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
    cost = _messages.MessageField('GoogleTypeMoney', 1)
    costInLocalCurrency = _messages.MessageField('GoogleTypeMoney', 2)
    duration = _messages.StringField(3)
    pricingType = _messages.EnumField('PricingTypeValueValuesEnum', 4)