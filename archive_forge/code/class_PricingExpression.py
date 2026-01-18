from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PricingExpression(_messages.Message):
    """Expresses a mathematical pricing formula. For Example:- `usage_unit:
  GBy` `tiered_rates:` `[start_usage_amount: 20, unit_price: $10]`
  `[start_usage_amount: 100, unit_price: $5]` The above expresses a pricing
  formula where the first 20GB is free, the next 80GB is priced at $10 per GB
  followed by $5 per GB for additional usage.

  Fields:
    baseUnit: The base unit for the SKU which is the unit used in usage
      exports. Example: "By"
    baseUnitConversionFactor: Conversion factor for converting from price per
      usage_unit to price per base_unit, and start_usage_amount to
      start_usage_amount in base_unit. unit_price /
      base_unit_conversion_factor = price per base_unit. start_usage_amount *
      base_unit_conversion_factor = start_usage_amount in base_unit.
    baseUnitDescription: The base unit in human readable form. Example:
      "byte".
    displayQuantity: The recommended quantity of units for displaying pricing
      info. When displaying pricing info it is recommended to display:
      (unit_price * display_quantity) per display_quantity usage_unit. This
      field does not affect the pricing formula and is for display purposes
      only. Example: If the unit_price is "0.0001 USD", the usage_unit is "GB"
      and the display_quantity is "1000" then the recommended way of
      displaying the pricing info is "0.10 USD per 1000 GB"
    tieredRates: The list of tiered rates for this pricing. The total cost is
      computed by applying each of the tiered rates on usage. This repeated
      list is sorted by ascending order of start_usage_amount.
    usageUnit: The short hand for unit of usage this pricing is specified in.
      Example: usage_unit of "GiBy" means that usage is specified in "Gibi
      Byte".
    usageUnitDescription: The unit of usage in human readable form. Example:
      "gibi byte".
  """
    baseUnit = _messages.StringField(1)
    baseUnitConversionFactor = _messages.FloatField(2)
    baseUnitDescription = _messages.StringField(3)
    displayQuantity = _messages.FloatField(4)
    tieredRates = _messages.MessageField('TierRate', 5, repeated=True)
    usageUnit = _messages.StringField(6)
    usageUnitDescription = _messages.StringField(7)