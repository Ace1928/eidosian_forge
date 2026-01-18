from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsSchedulesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsSchedulesListRequest object.

  Fields:
    filter: Lists the Schedules that match the filter expression. The
      following fields are supported: * `display_name`: Supports `=`, `!=`
      comparisons, and `:` wildcard. * `state`: Supports `=` and `!=`
      comparisons. * `request`: Supports existence of the check. (e.g.
      `create_pipeline_job_request:*` --> Schedule has
      create_pipeline_job_request). * `create_time`: Supports `=`, `!=`, `<`,
      `>`, `<=`, and `>=` comparisons. Values must be in RFC 3339 format. *
      `start_time`: Supports `=`, `!=`, `<`, `>`, `<=`, and `>=` comparisons.
      Values must be in RFC 3339 format. * `end_time`: Supports `=`, `!=`,
      `<`, `>`, `<=`, `>=` comparisons and `:*` existence check. Values must
      be in RFC 3339 format. * `next_run_time`: Supports `=`, `!=`, `<`, `>`,
      `<=`, and `>=` comparisons. Values must be in RFC 3339 format. Filter
      expressions can be combined together using logical operators (`NOT`,
      `AND` & `OR`). The syntax to define filter expression is based on
      https://google.aip.dev/160. Examples: * `state="ACTIVE" AND
      display_name:"my_schedule_*"` * `NOT display_name="my_schedule"` *
      `create_time>"2021-05-18T00:00:00Z"` * `end_time>"2021-05-18T00:00:00Z"
      OR NOT end_time:*` * `create_pipeline_job_request:*`
    orderBy: A comma-separated list of fields to order by. The default sort
      order is in ascending order. Use "desc" after a field name for
      descending. You can have multiple order_by fields provided. For example,
      using "create_time desc, end_time" will order results by create time in
      descending order, and if there are multiple schedules having the same
      create time, order them by the end time in ascending order. If order_by
      is not specified, it will order by default with create_time in
      descending order. Supported fields: * `create_time` * `start_time` *
      `end_time` * `next_run_time`
    pageSize: The standard list page size. Default to 100 if not specified.
    pageToken: The standard list page token. Typically obtained via
      ListSchedulesResponse.next_page_token of the previous
      ScheduleService.ListSchedules call.
    parent: Required. The resource name of the Location to list the Schedules
      from. Format: `projects/{project}/locations/{location}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)