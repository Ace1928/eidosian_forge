from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RatePlan(_messages.Message):
    """Rate plan details.

  Enums:
    BillingPeriodValueValuesEnum: Frequency at which the customer will be
      billed.
    ConsumptionPricingTypeValueValuesEnum: Pricing model used for consumption-
      based charges.
    PaymentFundingModelValueValuesEnum: DEPRECATED: This field is no longer
      supported and will eventually be removed when Apigee Hybrid 1.5/1.6 is
      no longer supported. Instead, use the `billingType` field inside
      `DeveloperMonetizationConfig` resource. Flag that specifies the billing
      account type, prepaid or postpaid.
    RevenueShareTypeValueValuesEnum: Method used to calculate the revenue that
      is shared with developers.
    StateValueValuesEnum: Current state of the rate plan (draft or published).

  Fields:
    apiproduct: Name of the API product that the rate plan is associated with.
    billingPeriod: Frequency at which the customer will be billed.
    consumptionPricingRates: API call volume ranges and the fees charged when
      the total number of API calls is within a given range. The method used
      to calculate the final fee depends on the selected pricing model. For
      example, if the pricing model is `STAIRSTEP` and the ranges are defined
      as follows: ``` { "start": 1, "end": 100, "fee": 75 }, { "start": 101,
      "end": 200, "fee": 100 }, } ``` Then the following fees would be charged
      based on the total number of API calls (assuming the currency selected
      is `USD`): * 1 call costs $75 * 50 calls cost $75 * 150 calls cost $100
      The number of API calls cannot exceed 200.
    consumptionPricingType: Pricing model used for consumption-based charges.
    createdAt: Output only. Time that the rate plan was created in
      milliseconds since epoch.
    currencyCode: Currency to be used for billing. Consists of a three-letter
      code as defined by the [ISO
      4217](https://en.wikipedia.org/wiki/ISO_4217) standard.
    description: Description of the rate plan.
    displayName: Display name of the rate plan.
    endTime: Time when the rate plan will expire in milliseconds since epoch.
      Set to 0 or `null` to indicate that the rate plan should never expire.
    fixedFeeFrequency: Frequency at which the fixed fee is charged.
    fixedRecurringFee: Fixed amount that is charged at a defined interval and
      billed in advance of use of the API product. The fee will be prorated
      for the first billing period.
    lastModifiedAt: Output only. Time the rate plan was last modified in
      milliseconds since epoch.
    name: Output only. Name of the rate plan.
    paymentFundingModel: DEPRECATED: This field is no longer supported and
      will eventually be removed when Apigee Hybrid 1.5/1.6 is no longer
      supported. Instead, use the `billingType` field inside
      `DeveloperMonetizationConfig` resource. Flag that specifies the billing
      account type, prepaid or postpaid.
    revenueShareRates: Details of the revenue sharing model.
    revenueShareType: Method used to calculate the revenue that is shared with
      developers.
    setupFee: Initial, one-time fee paid when purchasing the API product.
    startTime: Time when the rate plan becomes active in milliseconds since
      epoch.
    state: Current state of the rate plan (draft or published).
  """

    class BillingPeriodValueValuesEnum(_messages.Enum):
        """Frequency at which the customer will be billed.

    Values:
      BILLING_PERIOD_UNSPECIFIED: Billing period not specified.
      WEEKLY: Weekly billing period. **Note**: Not supported by Apigee at this
        time.
      MONTHLY: Monthly billing period.
    """
        BILLING_PERIOD_UNSPECIFIED = 0
        WEEKLY = 1
        MONTHLY = 2

    class ConsumptionPricingTypeValueValuesEnum(_messages.Enum):
        """Pricing model used for consumption-based charges.

    Values:
      CONSUMPTION_PRICING_TYPE_UNSPECIFIED: Pricing model not specified. This
        is the default.
      FIXED_PER_UNIT: Fixed rate charged for each API call.
      BANDED: Variable rate charged for each API call based on price tiers.
        Example: * 1-100 calls cost $2 per call * 101-200 calls cost $1.50 per
        call * 201-300 calls cost $1 per call * Total price for 50 calls: 50 x
        $2 = $100 * Total price for 150 calls: 100 x $2 + 50 x $1.5 = $275 *
        Total price for 250 calls: 100 x $2 + 100 x $1.5 + 50 x $1 = $400.
        **Note**: Not supported by Apigee at this time.
      TIERED: **Note**: Not supported by Apigee at this time.
      STAIRSTEP: **Note**: Not supported by Apigee at this time.
      BUNDLES: Cumulative rate charged for bundle of API calls whether or not
        the entire bundle is used. Example: * 1-100 calls cost $150 flat fee.
        * 101-200 calls cost $100 flat free. * 201-300 calls cost $75 flat
        fee. * Total price for 1 call: $150 * Total price for 50 calls: $150 *
        Total price for 150 calls: $150 + $100 * Total price for 250 calls:
        $150 + $100 + $75
    """
        CONSUMPTION_PRICING_TYPE_UNSPECIFIED = 0
        FIXED_PER_UNIT = 1
        BANDED = 2
        TIERED = 3
        STAIRSTEP = 4
        BUNDLES = 5

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

    class RevenueShareTypeValueValuesEnum(_messages.Enum):
        """Method used to calculate the revenue that is shared with developers.

    Values:
      REVENUE_SHARE_TYPE_UNSPECIFIED: Revenue share type is not specified.
      FIXED: Fixed percentage of the total revenue will be shared. The
        percentage to be shared can be configured by the API provider.
      VOLUME_BANDED: Amount of revenue shared depends on the number of API
        calls. The API call volume ranges and the revenue share percentage for
        each volume can be configured by the API provider. **Note**: Not
        supported by Apigee at this time.
    """
        REVENUE_SHARE_TYPE_UNSPECIFIED = 0
        FIXED = 1
        VOLUME_BANDED = 2

    class StateValueValuesEnum(_messages.Enum):
        """Current state of the rate plan (draft or published).

    Values:
      STATE_UNSPECIFIED: State of the rate plan is not specified.
      DRAFT: Rate plan is in draft mode and only visible to API providers.
      PUBLISHED: Rate plan is published and will become visible to developers
        for the configured duration (between `startTime` and `endTime`).
    """
        STATE_UNSPECIFIED = 0
        DRAFT = 1
        PUBLISHED = 2
    apiproduct = _messages.StringField(1)
    billingPeriod = _messages.EnumField('BillingPeriodValueValuesEnum', 2)
    consumptionPricingRates = _messages.MessageField('GoogleCloudApigeeV1RateRange', 3, repeated=True)
    consumptionPricingType = _messages.EnumField('ConsumptionPricingTypeValueValuesEnum', 4)
    createdAt = _messages.IntegerField(5)
    currencyCode = _messages.StringField(6)
    description = _messages.StringField(7)
    displayName = _messages.StringField(8)
    endTime = _messages.IntegerField(9)
    fixedFeeFrequency = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    fixedRecurringFee = _messages.MessageField('GoogleTypeMoney', 11)
    lastModifiedAt = _messages.IntegerField(12)
    name = _messages.StringField(13)
    paymentFundingModel = _messages.EnumField('PaymentFundingModelValueValuesEnum', 14)
    revenueShareRates = _messages.MessageField('GoogleCloudApigeeV1RevenueShareRange', 15, repeated=True)
    revenueShareType = _messages.EnumField('RevenueShareTypeValueValuesEnum', 16)
    setupFee = _messages.MessageField('GoogleTypeMoney', 17)
    startTime = _messages.IntegerField(18)
    state = _messages.EnumField('StateValueValuesEnum', 19)