from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1QueryTimeSeriesStatsRequest(_messages.Message):
    """QueryTimeSeriesStatsRequest represents a query that returns a collection
  of time series sequences grouped by their values.

  Enums:
    TimestampOrderValueValuesEnum: Order the sequences in increasing or
      decreasing order of timestamps. Default is descending order of
      timestamps (latest first).
    WindowSizeValueValuesEnum: Time buckets to group the stats by.

  Fields:
    dimensions: List of dimension names to group the aggregations by. If no
      dimensions are passed, a single trend line representing the requested
      metric aggregations grouped by environment is returned.
    filter: Filter further on specific dimension values. Follows the same
      grammar as custom report's filter expressions. Example, apiproxy eq
      'foobar'. https://cloud.google.com/apigee/docs/api-
      platform/analytics/analytics-reference#filters
    metrics: Required. List of metrics and their aggregations.
    pageSize: Page size represents the number of time series sequences, one
      per unique set of dimensions and their values.
    pageToken: Page token stands for a specific collection of time series
      sequences.
    timeRange: Required. Time range for the stats.
    timestampOrder: Order the sequences in increasing or decreasing order of
      timestamps. Default is descending order of timestamps (latest first).
    windowSize: Time buckets to group the stats by.
  """

    class TimestampOrderValueValuesEnum(_messages.Enum):
        """Order the sequences in increasing or decreasing order of timestamps.
    Default is descending order of timestamps (latest first).

    Values:
      ORDER_UNSPECIFIED: Unspecified order. Default is Descending.
      ASCENDING: Ascending sort order.
      DESCENDING: Descending sort order.
    """
        ORDER_UNSPECIFIED = 0
        ASCENDING = 1
        DESCENDING = 2

    class WindowSizeValueValuesEnum(_messages.Enum):
        """Time buckets to group the stats by.

    Values:
      WINDOW_SIZE_UNSPECIFIED: Unspecified window size. Default is 1 hour.
      MINUTE: 1 Minute window
      HOUR: 1 Hour window
      DAY: 1 Day window
      MONTH: 1 Month window
    """
        WINDOW_SIZE_UNSPECIFIED = 0
        MINUTE = 1
        HOUR = 2
        DAY = 3
        MONTH = 4
    dimensions = _messages.StringField(1, repeated=True)
    filter = _messages.StringField(2)
    metrics = _messages.MessageField('GoogleCloudApigeeV1MetricAggregation', 3, repeated=True)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    timeRange = _messages.MessageField('GoogleTypeInterval', 6)
    timestampOrder = _messages.EnumField('TimestampOrderValueValuesEnum', 7)
    windowSize = _messages.EnumField('WindowSizeValueValuesEnum', 8)