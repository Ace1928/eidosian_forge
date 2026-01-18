from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PricingPlanValueValuesEnum(_messages.Enum):
    """The pricing plan for this instance. This can be either `PER_USE` or
    `PACKAGE`. Only `PER_USE` is supported for Second Generation instances.

    Values:
      SQL_PRICING_PLAN_UNSPECIFIED: This is an unknown pricing plan for this
        instance.
      PACKAGE: The instance is billed at a monthly flat rate.
      PER_USE: The instance is billed per usage.
    """
    SQL_PRICING_PLAN_UNSPECIFIED = 0
    PACKAGE = 1
    PER_USE = 2