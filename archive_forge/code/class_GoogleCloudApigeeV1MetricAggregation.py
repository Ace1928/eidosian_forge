from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1MetricAggregation(_messages.Message):
    """The optionally aggregated metric to query with its ordering.

  Enums:
    AggregationValueValuesEnum: Aggregation function associated with the
      metric.
    OrderValueValuesEnum: Ordering for this aggregation in the result. For
      time series this is ignored since the ordering of points depends only on
      the timestamp, not the values.

  Fields:
    aggregation: Aggregation function associated with the metric.
    name: Name of the metric
    order: Ordering for this aggregation in the result. For time series this
      is ignored since the ordering of points depends only on the timestamp,
      not the values.
  """

    class AggregationValueValuesEnum(_messages.Enum):
        """Aggregation function associated with the metric.

    Values:
      AGGREGATION_FUNCTION_UNSPECIFIED: Unspecified Aggregation function.
      AVG: Average.
      SUM: Summation.
      MIN: Min.
      MAX: Max.
      COUNT_DISTINCT: Count distinct
    """
        AGGREGATION_FUNCTION_UNSPECIFIED = 0
        AVG = 1
        SUM = 2
        MIN = 3
        MAX = 4
        COUNT_DISTINCT = 5

    class OrderValueValuesEnum(_messages.Enum):
        """Ordering for this aggregation in the result. For time series this is
    ignored since the ordering of points depends only on the timestamp, not
    the values.

    Values:
      ORDER_UNSPECIFIED: Unspecified order. Default is Descending.
      ASCENDING: Ascending sort order.
      DESCENDING: Descending sort order.
    """
        ORDER_UNSPECIFIED = 0
        ASCENDING = 1
        DESCENDING = 2
    aggregation = _messages.EnumField('AggregationValueValuesEnum', 1)
    name = _messages.StringField(2)
    order = _messages.EnumField('OrderValueValuesEnum', 3)