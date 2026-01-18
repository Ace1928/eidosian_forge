from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TierRate(_messages.Message):
    """The price rate indicating starting usage and its corresponding price.

  Fields:
    startUsageAmount: Usage is priced at this rate only after this amount.
      Example: start_usage_amount of 10 indicates that the usage will be
      priced at the unit_price after the first 10 usage_units.
    unitPrice: The price per unit of usage. Example: unit_price of amount $10
      indicates that each unit will cost $10.
  """
    startUsageAmount = _messages.FloatField(1)
    unitPrice = _messages.MessageField('Money', 2)