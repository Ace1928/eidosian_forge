from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelDeploymentMonitoringJobsPauseRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelDeploymentMonitoringJobsPauseRequest
  object.

  Fields:
    googleCloudAiplatformV1PauseModelDeploymentMonitoringJobRequest: A
      GoogleCloudAiplatformV1PauseModelDeploymentMonitoringJobRequest resource
      to be passed as the request body.
    name: Required. The resource name of the ModelDeploymentMonitoringJob to
      pause. Format: `projects/{project}/locations/{location}/modelDeploymentM
      onitoringJobs/{model_deployment_monitoring_job}`
  """
    googleCloudAiplatformV1PauseModelDeploymentMonitoringJobRequest = _messages.MessageField('GoogleCloudAiplatformV1PauseModelDeploymentMonitoringJobRequest', 1)
    name = _messages.StringField(2, required=True)