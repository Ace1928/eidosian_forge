from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AggregationInfo(_messages.Message):
    """Represents the aggregation level and interval for pricing of a single
  SKU.

  Enums:
    AggregationIntervalValueValuesEnum:
    AggregationLevelValueValuesEnum:

  Fields:
    aggregationCount: The number of intervals to aggregate over. Example: If
      aggregation_level is "DAILY" and aggregation_count is 14, aggregation
      will be over 14 days.
    aggregationInterval: A AggregationIntervalValueValuesEnum attribute.
    aggregationLevel: A AggregationLevelValueValuesEnum attribute.
  """

    class AggregationIntervalValueValuesEnum(_messages.Enum):
        """AggregationIntervalValueValuesEnum enum type.

    Values:
      AGGREGATION_INTERVAL_UNSPECIFIED: <no description>
      DAILY: <no description>
      MONTHLY: <no description>
    """
        AGGREGATION_INTERVAL_UNSPECIFIED = 0
        DAILY = 1
        MONTHLY = 2

    class AggregationLevelValueValuesEnum(_messages.Enum):
        """AggregationLevelValueValuesEnum enum type.

    Values:
      AGGREGATION_LEVEL_UNSPECIFIED: <no description>
      ACCOUNT: <no description>
      PROJECT: <no description>
    """
        AGGREGATION_LEVEL_UNSPECIFIED = 0
        ACCOUNT = 1
        PROJECT = 2
    aggregationCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    aggregationInterval = _messages.EnumField('AggregationIntervalValueValuesEnum', 2)
    aggregationLevel = _messages.EnumField('AggregationLevelValueValuesEnum', 3)