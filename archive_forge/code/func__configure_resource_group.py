import json
import logging
import random
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable
from azure.common.credentials import get_cli_profile
from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode
def _configure_resource_group(config):
    subscription_id = config['provider'].get('subscription_id')
    if subscription_id is None:
        subscription_id = get_cli_profile().get_subscription_id()
    resource_client = ResourceManagementClient(AzureCliCredential(), subscription_id)
    config['provider']['subscription_id'] = subscription_id
    logger.info('Using subscription id: %s', subscription_id)
    assert 'resource_group' in config['provider'], 'Provider config must include resource_group field'
    resource_group = config['provider']['resource_group']
    assert 'location' in config['provider'], 'Provider config must include location field'
    params = {'location': config['provider']['location']}
    if 'tags' in config['provider']:
        params['tags'] = config['provider']['tags']
    logger.info('Creating/Updating resource group: %s', resource_group)
    rg_create_or_update = get_azure_sdk_function(client=resource_client.resource_groups, function_name='create_or_update')
    rg_create_or_update(resource_group_name=resource_group, parameters=params)
    current_path = Path(__file__).parent
    template_path = current_path.joinpath('azure-config-template.json')
    with open(template_path, 'r') as template_fp:
        template = json.load(template_fp)
    logger.info('Using cluster name: %s', config['cluster_name'])
    unique_id = config['provider'].get('unique_id')
    if unique_id is None:
        hasher = sha256()
        hasher.update(config['provider']['resource_group'].encode('utf-8'))
        unique_id = hasher.hexdigest()[:UNIQUE_ID_LEN]
    else:
        unique_id = str(unique_id)
    config['provider']['unique_id'] = unique_id
    logger.info('Using unique id: %s', unique_id)
    cluster_id = '{}-{}'.format(config['cluster_name'], unique_id)
    subnet_mask = config['provider'].get('subnet_mask')
    if subnet_mask is None:
        random.seed(unique_id)
        subnet_mask = '10.{}.0.0/16'.format(random.randint(1, 254))
    logger.info('Using subnet mask: %s', subnet_mask)
    parameters = {'properties': {'mode': DeploymentMode.incremental, 'template': template, 'parameters': {'subnet': {'value': subnet_mask}, 'clusterId': {'value': cluster_id}}}}
    create_or_update = get_azure_sdk_function(client=resource_client.deployments, function_name='create_or_update')
    outputs = create_or_update(resource_group_name=resource_group, deployment_name='ray-config', parameters=parameters).result().properties.outputs
    config['provider']['msi'] = outputs['msi']['value']
    config['provider']['nsg'] = outputs['nsg']['value']
    config['provider']['subnet'] = outputs['subnet']['value']
    return config