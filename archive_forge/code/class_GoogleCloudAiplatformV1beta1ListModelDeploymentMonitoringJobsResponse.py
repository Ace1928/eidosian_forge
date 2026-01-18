from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListModelDeploymentMonitoringJobsResponse(_messages.Message):
    """Response message for JobService.ListModelDeploymentMonitoringJobs.

  Fields:
    modelDeploymentMonitoringJobs: A list of ModelDeploymentMonitoringJobs
      that matches the specified filter in the request.
    nextPageToken: The standard List next-page token.
  """
    modelDeploymentMonitoringJobs = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelDeploymentMonitoringJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)