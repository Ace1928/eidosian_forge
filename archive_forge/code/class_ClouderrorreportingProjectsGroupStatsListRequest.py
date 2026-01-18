from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouderrorreportingProjectsGroupStatsListRequest(_messages.Message):
    """A ClouderrorreportingProjectsGroupStatsListRequest object.

  Enums:
    AlignmentValueValuesEnum: Optional. The alignment of the timed counts to
      be returned. Default is `ALIGNMENT_EQUAL_AT_END`.
    OrderValueValuesEnum: Optional. The sort order in which the results are
      returned. Default is `COUNT_DESC`.
    TimeRangePeriodValueValuesEnum: Restricts the query to the specified time
      range.

  Fields:
    alignment: Optional. The alignment of the timed counts to be returned.
      Default is `ALIGNMENT_EQUAL_AT_END`.
    alignmentTime: Optional. Time where the timed counts shall be aligned if
      rounded alignment is chosen. Default is 00:00 UTC.
    groupId: Optional. List all ErrorGroupStats with these IDs. The `group_id`
      is a unique identifier for a particular error group. The identifier is
      derived from key parts of the error-log content and is treated as
      Service Data. For information about how Service Data is handled, see
      [Google Cloud Privacy Notice] (https://cloud.google.com/terms/cloud-
      privacy-notice).
    order: Optional. The sort order in which the results are returned. Default
      is `COUNT_DESC`.
    pageSize: Optional. The maximum number of results to return per response.
      Default is 20.
    pageToken: Optional. A next_page_token provided by a previous response. To
      view additional results, pass this token along with the identical query
      parameters as the first request.
    projectName: Required. The resource name of the Google Cloud Platform
      project. Written as `projects/{projectID}` or
      `projects/{projectNumber}`, where `{projectID}` and `{projectNumber}`
      can be found in the [Google Cloud
      console](https://support.google.com/cloud/answer/6158840). Examples:
      `projects/my-project-123`, `projects/5551234`.
    serviceFilter_resourceType: Optional. The exact value to match against
      [`ServiceContext.resource_type`](/error-
      reporting/reference/rest/v1beta1/ServiceContext#FIELDS.resource_type).
    serviceFilter_service: Optional. The exact value to match against
      [`ServiceContext.service`](/error-
      reporting/reference/rest/v1beta1/ServiceContext#FIELDS.service).
    serviceFilter_version: Optional. The exact value to match against
      [`ServiceContext.version`](/error-
      reporting/reference/rest/v1beta1/ServiceContext#FIELDS.version).
    timeRange_period: Restricts the query to the specified time range.
    timedCountDuration: Optional. The preferred duration for a single returned
      TimedCount. If not set, no timed counts are returned.
  """

    class AlignmentValueValuesEnum(_messages.Enum):
        """Optional. The alignment of the timed counts to be returned. Default is
    `ALIGNMENT_EQUAL_AT_END`.

    Values:
      ERROR_COUNT_ALIGNMENT_UNSPECIFIED: No alignment specified.
      ALIGNMENT_EQUAL_ROUNDED: The time periods shall be consecutive, have
        width equal to the requested duration, and be aligned at the
        alignment_time provided in the request. The alignment_time does not
        have to be inside the query period but even if it is outside, only
        time periods are returned which overlap with the query period. A
        rounded alignment will typically result in a different size of the
        first or the last time period.
      ALIGNMENT_EQUAL_AT_END: The time periods shall be consecutive, have
        width equal to the requested duration, and be aligned at the end of
        the requested time period. This can result in a different size of the
        first time period.
    """
        ERROR_COUNT_ALIGNMENT_UNSPECIFIED = 0
        ALIGNMENT_EQUAL_ROUNDED = 1
        ALIGNMENT_EQUAL_AT_END = 2

    class OrderValueValuesEnum(_messages.Enum):
        """Optional. The sort order in which the results are returned. Default is
    `COUNT_DESC`.

    Values:
      GROUP_ORDER_UNSPECIFIED: No group order specified.
      COUNT_DESC: Total count of errors in the given time window in descending
        order.
      LAST_SEEN_DESC: Timestamp when the group was last seen in the given time
        window in descending order.
      CREATED_DESC: Timestamp when the group was created in descending order.
      AFFECTED_USERS_DESC: Number of affected users in the given time window
        in descending order.
    """
        GROUP_ORDER_UNSPECIFIED = 0
        COUNT_DESC = 1
        LAST_SEEN_DESC = 2
        CREATED_DESC = 3
        AFFECTED_USERS_DESC = 4

    class TimeRangePeriodValueValuesEnum(_messages.Enum):
        """Restricts the query to the specified time range.

    Values:
      PERIOD_UNSPECIFIED: Do not use.
      PERIOD_1_HOUR: Retrieve data for the last hour. Recommended minimum
        timed count duration: 1 min.
      PERIOD_6_HOURS: Retrieve data for the last 6 hours. Recommended minimum
        timed count duration: 10 min.
      PERIOD_1_DAY: Retrieve data for the last day. Recommended minimum timed
        count duration: 1 hour.
      PERIOD_1_WEEK: Retrieve data for the last week. Recommended minimum
        timed count duration: 6 hours.
      PERIOD_30_DAYS: Retrieve data for the last 30 days. Recommended minimum
        timed count duration: 1 day.
    """
        PERIOD_UNSPECIFIED = 0
        PERIOD_1_HOUR = 1
        PERIOD_6_HOURS = 2
        PERIOD_1_DAY = 3
        PERIOD_1_WEEK = 4
        PERIOD_30_DAYS = 5
    alignment = _messages.EnumField('AlignmentValueValuesEnum', 1)
    alignmentTime = _messages.StringField(2)
    groupId = _messages.StringField(3, repeated=True)
    order = _messages.EnumField('OrderValueValuesEnum', 4)
    pageSize = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(6)
    projectName = _messages.StringField(7, required=True)
    serviceFilter_resourceType = _messages.StringField(8)
    serviceFilter_service = _messages.StringField(9)
    serviceFilter_version = _messages.StringField(10)
    timeRange_period = _messages.EnumField('TimeRangePeriodValueValuesEnum', 11)
    timedCountDuration = _messages.StringField(12)