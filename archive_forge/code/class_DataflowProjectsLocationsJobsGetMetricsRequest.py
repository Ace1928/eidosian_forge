from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsJobsGetMetricsRequest(_messages.Message):
    """A DataflowProjectsLocationsJobsGetMetricsRequest object.

  Fields:
    jobId: The job to get metrics for.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the job specified by job_id.
    projectId: A project id.
    startTime: Return only metric data that has changed since this time.
      Default is to return all information about all metrics for the job.
  """
    jobId = _messages.StringField(1, required=True)
    location = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
    startTime = _messages.StringField(4)