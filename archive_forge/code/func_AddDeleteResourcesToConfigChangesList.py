import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def AddDeleteResourcesToConfigChangesList(delete_resources_list, config_changes_list, enum, api_version):
    """Adds the resources having update type as delete to the config_changes_list."""
    for resource_self_link in delete_resources_list:
        resource_config = {'selfLink': resource_self_link}
        config_change = Messages(api_version).ConfigChange(updateType=enum.DELETE, assetType='compute.googleapis.com/Firewall', proposedConfigBody=json.dumps(resource_config))
        config_changes_list.append(config_change)
    return config_changes_list