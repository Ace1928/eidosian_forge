from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsJobsGetExecutionDetailsRequest(_messages.Message):
    """A DataflowProjectsLocationsJobsGetExecutionDetailsRequest object.

  Fields:
    jobId: The job to get execution details for.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the job specified by job_id.
    pageSize: If specified, determines the maximum number of stages to return.
      If unspecified, the service may choose an appropriate default, or may
      return an arbitrarily large number of results.
    pageToken: If supplied, this should be the value of next_page_token
      returned by an earlier call. This will cause the next page of results to
      be returned.
    projectId: A project id.
  """
    jobId = _messages.StringField(1, required=True)
    location = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    projectId = _messages.StringField(5, required=True)