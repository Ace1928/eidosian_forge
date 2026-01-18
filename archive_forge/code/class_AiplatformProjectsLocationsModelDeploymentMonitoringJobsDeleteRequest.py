from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelDeploymentMonitoringJobsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelDeploymentMonitoringJobsDeleteRequest
  object.

  Fields:
    name: Required. The resource name of the model monitoring job to delete.
      Format: `projects/{project}/locations/{location}/modelDeploymentMonitori
      ngJobs/{model_deployment_monitoring_job}`
  """
    name = _messages.StringField(1, required=True)