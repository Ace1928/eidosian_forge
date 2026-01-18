import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def ParseAndAddResourceConfigToConfigChangesList(gcp_config, tainted_resources_list, update_resource_list, enum, api_version):
    """Parses the resources from gcp_config file and adds them to config_changes_list."""
    config_changes_list = []
    for resource_object in gcp_config['resource_body']:
        if 'asset_type' not in resource_object:
            raise InvalidInputError('Error parsing config changes file.')
        asset_type = resource_object['asset_type']
        self_link = resource_object['name'].replace('//compute.googleapis.com', 'https://www.googleapis.com/compute/v1')
        if asset_type != 'compute.googleapis.com/Firewall' or self_link in tainted_resources_list:
            continue
        proposed_resource_config = resource_object['resource']['data']
        proposed_resource_config['kind'] = 'compute#firewall'
        if self_link in update_resource_list:
            update_type = enum.UPDATE
            proposed_resource_config['selfLink'] = self_link
        else:
            update_type = enum.INSERT
            proposed_resource_config['selfLink'] = self_link
            if 'direction' not in proposed_resource_config:
                proposed_resource_config['direction'] = 'INGRESS'
        config_change = Messages(api_version).ConfigChange(updateType=update_type, assetType=asset_type, proposedConfigBody=json.dumps(proposed_resource_config))
        config_changes_list.append(config_change)
    return config_changes_list