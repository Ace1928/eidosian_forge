import json
from googlecloudsdk.api_lib.network_management.simulation import Messages
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
def IdentifyChangeUpdateType(proposed_resource_config, original_resource_config_list, api_version, update_resource_list):
    """Given a proposed resource config, it returns the update type."""
    enum = Messages(api_version).ConfigChange.UpdateTypeValueValuesEnum
    if 'selfLink' in proposed_resource_config:
        self_link = proposed_resource_config['selfLink']
        if self_link in original_resource_config_list:
            update_resource_list.append(self_link)
            return enum.UPDATE
    else:
        AddSelfLinkGCPResource(proposed_resource_config)
        return enum.INSERT