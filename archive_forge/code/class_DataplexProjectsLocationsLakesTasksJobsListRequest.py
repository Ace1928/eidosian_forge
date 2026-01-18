from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesTasksJobsListRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesTasksJobsListRequest object.

  Fields:
    pageSize: Optional. Maximum number of jobs to return. The service may
      return fewer than this value. If unspecified, at most 10 jobs will be
      returned. The maximum value is 1000; values above 1000 will be coerced
      to 1000.
    pageToken: Optional. Page token received from a previous ListJobs call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to ListJobs must match the call that provided the
      page token.
    parent: Required. The resource name of the parent environment: projects/{p
      roject_number}/locations/{location_id}/lakes/{lake_id}/tasks/{task_id}.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)