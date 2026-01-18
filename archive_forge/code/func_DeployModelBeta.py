from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.api_lib.ai.models import client as model_client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.credentials import requests
from six.moves import http_client
def DeployModelBeta(self, endpoint_ref, model, region, display_name, machine_type=None, accelerator_dict=None, min_replica_count=None, max_replica_count=None, autoscaling_metric_specs=None, enable_access_logging=False, enable_container_logging=False, service_account=None, traffic_split=None, deployed_model_id=None, shared_resources_ref=None):
    """Deploys a model to an existing endpoint using v1beta1 API.

    Args:
      endpoint_ref: Resource, the parsed endpoint that the model is deployed to.
      model: str, Id of the uploaded model to be deployed.
      region: str, the location of the endpoint and the model.
      display_name: str, the display name of the new deployed model.
      machine_type: str or None, the type of the machine to serve the model.
      accelerator_dict: dict or None, the accelerator attached to the deployed
        model from args.
      min_replica_count: int or None, the minimum number of replicas the
        deployed model will be always deployed on.
      max_replica_count: int or None, the maximum number of replicas the
        deployed model may be deployed on.
      autoscaling_metric_specs: dict or None, the metric specification that
        defines the target resource utilization for calculating the desired
        replica count.
      enable_access_logging: bool, whether or not enable access logs.
      enable_container_logging: bool, whether or not enable container logging.
      service_account: str or None, the service account that the deployed model
        runs as.
      traffic_split: dict or None, the new traffic split of the endpoint.
      deployed_model_id: str or None, id of the deployed model.
      shared_resources_ref: str or None, the shared deployment resource pool the
        model should use

    Returns:
      A long-running operation for DeployModel.
    """
    model_ref = _ParseModel(model, region)
    resource_type = _GetModelDeploymentResourceType(model_ref, self.client, shared_resources_ref)
    deployed_model = None
    if resource_type == 'DEDICATED_RESOURCES':
        machine_spec = self.messages.GoogleCloudAiplatformV1beta1MachineSpec()
        if machine_type is not None:
            machine_spec.machineType = machine_type
        accelerator = flags.ParseAcceleratorFlag(accelerator_dict, constants.BETA_VERSION)
        if accelerator is not None:
            machine_spec.acceleratorType = accelerator.acceleratorType
            machine_spec.acceleratorCount = accelerator.acceleratorCount
        dedicated = self.messages.GoogleCloudAiplatformV1beta1DedicatedResources(machineSpec=machine_spec)
        dedicated.minReplicaCount = min_replica_count or 1
        if max_replica_count is not None:
            dedicated.maxReplicaCount = max_replica_count
        if autoscaling_metric_specs is not None:
            autoscaling_metric_specs_list = []
            for name, target in sorted(autoscaling_metric_specs.items()):
                autoscaling_metric_specs_list.append(self.messages.GoogleCloudAiplatformV1beta1AutoscalingMetricSpec(metricName=constants.OP_AUTOSCALING_METRIC_NAME_MAPPER[name], target=target))
            dedicated.autoscalingMetricSpecs = autoscaling_metric_specs_list
        deployed_model = self.messages.GoogleCloudAiplatformV1beta1DeployedModel(dedicatedResources=dedicated, displayName=display_name, model=model_ref.RelativeName())
    elif resource_type == 'AUTOMATIC_RESOURCES':
        automatic = self.messages.GoogleCloudAiplatformV1beta1AutomaticResources()
        if min_replica_count is not None:
            automatic.minReplicaCount = min_replica_count
        if max_replica_count is not None:
            automatic.maxReplicaCount = max_replica_count
        deployed_model = self.messages.GoogleCloudAiplatformV1beta1DeployedModel(automaticResources=automatic, displayName=display_name, model=model_ref.RelativeName())
    else:
        deployed_model = self.messages.GoogleCloudAiplatformV1beta1DeployedModel(displayName=display_name, model=model_ref.RelativeName(), sharedResources=shared_resources_ref.RelativeName())
    deployed_model.enableAccessLogging = enable_access_logging
    deployed_model.enableContainerLogging = enable_container_logging
    if service_account is not None:
        deployed_model.serviceAccount = service_account
    if deployed_model_id is not None:
        deployed_model.id = deployed_model_id
    deployed_model_req = self.messages.GoogleCloudAiplatformV1beta1DeployModelRequest(deployedModel=deployed_model)
    if traffic_split is not None:
        additional_properties = []
        for key, value in sorted(traffic_split.items()):
            additional_properties.append(deployed_model_req.TrafficSplitValue().AdditionalProperty(key=key, value=value))
        deployed_model_req.trafficSplit = deployed_model_req.TrafficSplitValue(additionalProperties=additional_properties)
    req = self.messages.AiplatformProjectsLocationsEndpointsDeployModelRequest(endpoint=endpoint_ref.RelativeName(), googleCloudAiplatformV1beta1DeployModelRequest=deployed_model_req)
    return self.client.projects_locations_endpoints.DeployModel(req)