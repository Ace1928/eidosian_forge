from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1QueryTabularStatsRequest(_messages.Message):
    """Request payload representing the query to be run for fetching security
  statistics as rows.

  Fields:
    dimensions: Required. List of dimension names to group the aggregations
      by.
    filter: Filter further on specific dimension values. Follows the same
      grammar as custom report's filter expressions. Example, apiproxy eq
      'foobar'. https://cloud.google.com/apigee/docs/api-
      platform/analytics/analytics-reference#filters
    metrics: Required. List of metrics and their aggregations.
    pageSize: Page size represents the number of rows.
    pageToken: Identifies a sequence of rows.
    timeRange: Time range for the stats.
  """
    dimensions = _messages.StringField(1, repeated=True)
    filter = _messages.StringField(2)
    metrics = _messages.MessageField('GoogleCloudApigeeV1MetricAggregation', 3, repeated=True)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    timeRange = _messages.MessageField('GoogleTypeInterval', 6)