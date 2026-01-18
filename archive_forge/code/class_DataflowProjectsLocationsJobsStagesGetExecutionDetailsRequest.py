from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsJobsStagesGetExecutionDetailsRequest(_messages.Message):
    """A DataflowProjectsLocationsJobsStagesGetExecutionDetailsRequest object.

  Fields:
    endTime: Upper time bound of work items to include, by start time.
    jobId: The job to get execution details for.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the job specified by job_id.
    pageSize: If specified, determines the maximum number of work items to
      return. If unspecified, the service may choose an appropriate default,
      or may return an arbitrarily large number of results.
    pageToken: If supplied, this should be the value of next_page_token
      returned by an earlier call. This will cause the next page of results to
      be returned.
    projectId: A project id.
    stageId: The stage for which to fetch information.
    startTime: Lower time bound of work items to include, by start time.
  """
    endTime = _messages.StringField(1)
    jobId = _messages.StringField(2, required=True)
    location = _messages.StringField(3, required=True)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    projectId = _messages.StringField(6, required=True)
    stageId = _messages.StringField(7, required=True)
    startTime = _messages.StringField(8)