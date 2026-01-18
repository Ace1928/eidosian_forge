from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Query(_messages.Message):
    """A GoogleCloudApigeeV1Query object.

  Fields:
    csvDelimiter: Delimiter used in the CSV file, if `outputFormat` is set to
      `csv`. Defaults to the `,` (comma) character. Supported delimiter
      characters include comma (`,`), pipe (`|`), and tab (`\\t`).
    dimensions: A list of dimensions. https://docs.apigee.com/api-
      platform/analytics/analytics-reference#dimensions
    envgroupHostname: Hostname needs to be specified if query intends to run
      at host level. This field is only allowed when query is submitted by
      CreateHostAsyncQuery where analytics data will be grouped by
      organization and hostname.
    filter: Boolean expression that can be used to filter data. Filter
      expressions can be combined using AND/OR terms and should be fully
      parenthesized to avoid ambiguity. See Analytics metrics, dimensions, and
      filters reference https://docs.apigee.com/api-
      platform/analytics/analytics-reference for more information on the
      fields available to filter on. For more information on the tokens that
      you use to build filter expressions, see Filter expression syntax.
      https://docs.apigee.com/api-platform/analytics/asynch-reports-
      api#filter-expression-syntax
    groupByTimeUnit: Time unit used to group the result set. Valid values
      include: second, minute, hour, day, week, or month. If a query includes
      groupByTimeUnit, then the result is an aggregation based on the
      specified time unit and the resultant timestamp does not include
      milliseconds precision. If a query omits groupByTimeUnit, then the
      resultant timestamp includes milliseconds precision.
    limit: Maximum number of rows that can be returned in the result.
    metrics: A list of Metrics.
    name: Asynchronous Query Name.
    outputFormat: Valid values include: `csv` or `json`. Defaults to `json`.
      Note: Configure the delimiter for CSV output using the csvDelimiter
      property.
    reportDefinitionId: Asynchronous Report ID.
    timeRange: Required. Time range for the query. Can use the following
      predefined strings to specify the time range: `last60minutes`
      `last24hours` `last7days` Or, specify the timeRange as a structure
      describing start and end timestamps in the ISO format: yyyy-mm-
      ddThh:mm:ssZ. Example: "timeRange": { "start": "2018-07-29T00:13:00Z",
      "end": "2018-08-01T00:18:00Z" }
  """
    csvDelimiter = _messages.StringField(1)
    dimensions = _messages.StringField(2, repeated=True)
    envgroupHostname = _messages.StringField(3)
    filter = _messages.StringField(4)
    groupByTimeUnit = _messages.StringField(5)
    limit = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    metrics = _messages.MessageField('GoogleCloudApigeeV1QueryMetric', 7, repeated=True)
    name = _messages.StringField(8)
    outputFormat = _messages.StringField(9)
    reportDefinitionId = _messages.StringField(10)
    timeRange = _messages.MessageField('extra_types.JsonValue', 11)