from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Money(_messages.Message):
    """Represents an amount of money with its currency type.

  Fields:
    currencyCode: The three-letter currency code defined in ISO 4217.
    nanos: Number of nano (10^-9) units of the amount. The value must be
      between -999,999,999 and +999,999,999 inclusive. If `units` is positive,
      `nanos` must be positive or zero. If `units` is zero, `nanos` can be
      positive, zero, or negative. If `units` is negative, `nanos` must be
      negative or zero. For example $-1.75 is represented as `units`=-1 and
      `nanos`=-750,000,000.
    units: The whole units of the amount. For example if `currencyCode` is
      `"USD"`, then 1 unit is one US dollar.
  """
    currencyCode = _messages.StringField(1)
    nanos = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    units = _messages.IntegerField(3)