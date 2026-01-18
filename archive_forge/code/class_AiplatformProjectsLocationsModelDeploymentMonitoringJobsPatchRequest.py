from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelDeploymentMonitoringJobsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelDeploymentMonitoringJobsPatchRequest
  object.

  Fields:
    googleCloudAiplatformV1ModelDeploymentMonitoringJob: A
      GoogleCloudAiplatformV1ModelDeploymentMonitoringJob resource to be
      passed as the request body.
    name: Output only. Resource name of a ModelDeploymentMonitoringJob.
    updateMask: Required. The update mask is used to specify the fields to be
      overwritten in the ModelDeploymentMonitoringJob resource by the update.
      The fields specified in the update_mask are relative to the resource,
      not the full request. A field will be overwritten if it is in the mask.
      If the user does not provide a mask then only the non-empty fields
      present in the request will be overwritten. Set the update_mask to `*`
      to override all fields. For the objective config, the user can either
      provide the update mask for
      model_deployment_monitoring_objective_configs or any combination of its
      nested fields, such as: model_deployment_monitoring_objective_configs.ob
      jective_config.training_dataset. Updatable fields: * `display_name` *
      `model_deployment_monitoring_schedule_config` *
      `model_monitoring_alert_config` * `logging_sampling_strategy` * `labels`
      * `log_ttl` * `enable_monitoring_pipeline_logs` . and *
      `model_deployment_monitoring_objective_configs` . or * `model_deployment
      _monitoring_objective_configs.objective_config.training_dataset` * `mode
      l_deployment_monitoring_objective_configs.objective_config.training_pred
      iction_skew_detection_config` * `model_deployment_monitoring_objective_c
      onfigs.objective_config.prediction_drift_detection_config`
  """
    googleCloudAiplatformV1ModelDeploymentMonitoringJob = _messages.MessageField('GoogleCloudAiplatformV1ModelDeploymentMonitoringJob', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)