from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import custom_target_type_util
from googlecloudsdk.command_lib.deploy import deploy_policy_util
from googlecloudsdk.command_lib.deploy import manifest_util
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.core import log
def CreateResources(self, manifests, region):
    """Creates Cloud Deploy resources.

    Asynchronously calls the API then iterate the operations
    to check the status.

    Args:
     manifests: the list of parsed resource yaml definitions.
     region: location ID.
    """
    resource_dict = manifest_util.ParseDeployConfig(self.messages, manifests, region)
    msg_template = 'Created Cloud Deploy resource: {}.'
    pipelines = resource_dict[manifest_util.DELIVERY_PIPELINE_KIND_V1BETA1]
    if pipelines:
        operation_dict = {}
        for resource in pipelines:
            operation_dict[resource.name] = self.CreateDeliveryPipeline(resource)
        self.operation_client.CheckOperationStatus(operation_dict, msg_template)
    targets = resource_dict[manifest_util.TARGET_KIND_V1BETA1]
    if targets:
        operation_dict = {}
        for resource in targets:
            operation_dict[resource.name] = target_util.PatchTarget(resource)
        self.operation_client.CheckOperationStatus(operation_dict, msg_template)
    automations = resource_dict[manifest_util.AUTOMATION_KIND]
    operation_dict = {}
    for resource in automations:
        operation_dict[resource.name] = automation_util.PatchAutomation(resource)
    self.operation_client.CheckOperationStatus(operation_dict, msg_template)
    custom_target_types = resource_dict[manifest_util.CUSTOM_TARGET_TYPE_KIND]
    operation_dict = {}
    for resource in custom_target_types:
        operation_dict[resource.name] = custom_target_type_util.PatchCustomTargetType(resource)
    self.operation_client.CheckOperationStatus(operation_dict, msg_template)
    deploy_policies = resource_dict[manifest_util.DEPLOY_POLICY_KIND]
    operation_dict = {}
    for resource in deploy_policies:
        operation_dict[resource.name] = deploy_policy_util.PatchDeployPolicy(resource)
    self.operation_client.CheckOperationStatus(operation_dict, msg_template)