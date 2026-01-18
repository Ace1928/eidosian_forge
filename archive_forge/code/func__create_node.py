import json
import logging
from pathlib import Path
from threading import RLock
from uuid import uuid4
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode
from ray.autoscaler._private._azure.config import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _create_node(self, node_config, tags, count):
    """Creates a number of nodes within the namespace."""
    resource_group = self.provider_config['resource_group']
    current_path = Path(__file__).parent
    template_path = current_path.joinpath('azure-vm-template.json')
    with open(template_path, 'r') as template_fp:
        template = json.load(template_fp)
    config_tags = node_config.get('tags', {}).copy()
    config_tags.update(tags)
    config_tags[TAG_RAY_CLUSTER_NAME] = self.cluster_name
    vm_name = '{node}-{unique_id}-{vm_id}'.format(node=config_tags.get(TAG_RAY_NODE_NAME, 'node'), unique_id=self.provider_config['unique_id'], vm_id=uuid4().hex[:UNIQUE_ID_LEN])[:VM_NAME_MAX_LEN]
    use_internal_ips = self.provider_config.get('use_internal_ips', False)
    template_params = node_config['azure_arm_parameters'].copy()
    template_params['vmName'] = vm_name
    template_params['provisionPublicIp'] = not use_internal_ips
    template_params['vmTags'] = config_tags
    template_params['vmCount'] = count
    template_params['msi'] = self.provider_config['msi']
    template_params['nsg'] = self.provider_config['nsg']
    template_params['subnet'] = self.provider_config['subnet']
    parameters = {'properties': {'mode': DeploymentMode.incremental, 'template': template, 'parameters': {key: {'value': value} for key, value in template_params.items()}}}
    create_or_update = get_azure_sdk_function(client=self.resource_client.deployments, function_name='create_or_update')
    create_or_update(resource_group_name=resource_group, deployment_name=vm_name, parameters=parameters).wait()