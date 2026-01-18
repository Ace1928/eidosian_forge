from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsSnoozesListRequest(_messages.Message):
    """A MonitoringProjectsSnoozesListRequest object.

  Fields:
    filter: Optional. Optional filter to restrict results to the given
      criteria. The following fields are supported. interval.start_time
      interval.end_timeFor example: ``` interval.start_time >
      "2022-03-11T00:00:00-08:00" AND interval.end_time <
      "2022-03-12T00:00:00-08:00" ```
    pageSize: Optional. The maximum number of results to return for a single
      query. The server may further constrain the maximum number of results
      returned in a single page. The value should be in the range 1, 1000. If
      the value given is outside this range, the server will decide the number
      of results to be returned.
    pageToken: Optional. The next_page_token from a previous call to
      ListSnoozesRequest to get the next page of results.
    parent: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) whose Snoozes
      should be listed. The format is: projects/[PROJECT_ID_OR_NUMBER]
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)