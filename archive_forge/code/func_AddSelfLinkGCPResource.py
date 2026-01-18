import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def AddSelfLinkGCPResource(proposed_resource_config):
    if 'name' not in proposed_resource_config:
        raise InvalidInputError('`name` key missing in one of resource(s) config.')
    name = proposed_resource_config['name']
    project_id = properties.VALUES.core.project.Get()
    proposed_resource_config['selfLink'] = 'projects/{}/global/firewalls/{}'.format(project_id, name)