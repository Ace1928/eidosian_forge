from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsHostStatsGetRequest(_messages.Message):
    """A ApigeeOrganizationsHostStatsGetRequest object.

  Fields:
    accuracy: No longer used by Apigee. Supported for backwards compatibility.
    envgroupHostname: Required. Hostname for which the interactive query will
      be executed.
    filter: Flag that enables drill-down on specific dimension values.
    limit: Maximum number of result items to return. The default and maximum
      value that can be returned is 14400.
    name: Required. Resource name for which the interactive query will be
      executed. Use the following format in your request:
      `organizations/{org}/hostStats/{dimensions}` Dimensions let you view
      metrics in meaningful groupings, such as `apiproxy`, `target_host`. The
      value of dimensions should be a comma-separated list as shown below
      `organizations/{org}/hostStats/apiproxy,request_verb`
    offset: Offset value. Use `offset` with `limit` to enable pagination of
      results. For example, to display results 11-20, set limit to `10` and
      offset to `10`.
    realtime: No longer used by Apigee. Supported for backwards compatibility.
    select: Comma-separated list of metrics. For example:
      `sum(message_count),sum(error_count)`
    sort: Flag that specifies if the sort order should be ascending or
      descending. Valid values are `DESC` and `ASC`.
    sortby: Comma-separated list of columns to sort the final result.
    timeRange: Time interval for the interactive query. Time range is
      specified in GMT as `start~end`. For example: `04/15/2017
      00:00~05/15/2017 23:59`
    timeUnit: Granularity of metrics returned. Valid values include: `second`,
      `minute`, `hour`, `day`, `week`, or `month`.
    topk: Top number of results to return. For example, to return the top 5
      results, set `topk=5`.
    tsAscending: Flag that specifies whether to list timestamps in ascending
      (`true`) or descending (`false`) order. Apigee recommends that you set
      this value to `true` if you are using `sortby` with `sort=DESC`.
    tzo: Timezone offset value.
  """
    accuracy = _messages.StringField(1)
    envgroupHostname = _messages.StringField(2)
    filter = _messages.StringField(3)
    limit = _messages.StringField(4)
    name = _messages.StringField(5, required=True)
    offset = _messages.StringField(6)
    realtime = _messages.BooleanField(7)
    select = _messages.StringField(8)
    sort = _messages.StringField(9)
    sortby = _messages.StringField(10)
    timeRange = _messages.StringField(11)
    timeUnit = _messages.StringField(12)
    topk = _messages.StringField(13)
    tsAscending = _messages.BooleanField(14)
    tzo = _messages.StringField(15)