from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelDeploymentMonitoringJobsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelDeploymentMonitoringJobsGetRequest
  object.

  Fields:
    name: Required. The resource name of the ModelDeploymentMonitoringJob.
      Format: `projects/{project}/locations/{location}/modelDeploymentMonitori
      ngJobs/{model_deployment_monitoring_job}`
  """
    name = _messages.StringField(1, required=True)