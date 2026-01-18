import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.command_lib.scc.manage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
def CreateUpdateMaskFromArgs(args):
    """Create an update mask with the args given."""
    if args.enablement_state is not None and args.custom_config_file is not None:
        return 'enablement_state,custom_config'
    elif args.enablement_state is not None:
        return 'enablement_state'
    elif args.custom_config_file is not None:
        return 'custom_config'
    else:
        raise errors.InvalidUpdateMaskInputError('Error parsing Update Mask. Either a custom configuration or an enablement state (or both) must be provided to update the custom module.')