import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def ParseGCPSimulationConfigChangesFile(proposed_changes_file, api_version, simulation_type, original_config_file):
    """Parse and convert the config changes file into API Format."""
    try:
        proposed_resources_config = yaml.load_path(proposed_changes_file)
    except yaml.YAMLParseError as unused_ref:
        raise InvalidFileError('Error parsing config changes file: [{}]'.format(proposed_changes_file))
    try:
        original_resources_config = yaml.load_path(original_config_file)
    except yaml.YAMLParseError as unused_ref:
        raise InvalidFileError('Error parsing the original config file: [{}]'.format(original_config_file))
    original_config_resource_list = []
    update_resource_list = []
    config_changes_list = []
    for original_resource_config in original_resources_config:
        if 'kind' not in original_resource_config:
            raise InvalidInputError('`kind` key missing in one of resource(s) config.')
        if 'selfLink' not in original_resource_config:
            raise InvalidInputError('`selfLink` missing in one of original resource(s) config.')
        original_config_resource_list.append(original_resource_config['selfLink'])
    for proposed_resource_config in proposed_resources_config:
        if 'kind' not in proposed_resource_config:
            raise InvalidInputError('`kind` key missing in one of resource(s) config.')
        if 'direction' not in proposed_resource_config:
            proposed_resource_config['direction'] = 'INGRESS'
        update_type = IdentifyChangeUpdateType(proposed_resource_config, original_config_resource_list, api_version, update_resource_list)
        config_change = Messages(api_version).ConfigChange(updateType=update_type, assetType=MapResourceType(proposed_resource_config['kind']), proposedConfigBody=json.dumps(proposed_resource_config, sort_keys=True))
        config_changes_list.append(config_change)
    enum = Messages(api_version).ConfigChange.UpdateTypeValueValuesEnum
    for original_resource_config in original_resources_config:
        original_self_link = original_resource_config['selfLink']
        if original_self_link not in update_resource_list:
            resource_config = {'selfLink': original_self_link}
            config_change = Messages(api_version).ConfigChange(updateType=enum.DELETE, assetType=MapResourceType(original_resource_config['kind']), proposedConfigBody=json.dumps(resource_config, sort_keys=True))
            config_changes_list.append(config_change)
    return MapSimulationTypeToRequest(api_version, config_changes_list, simulation_type)