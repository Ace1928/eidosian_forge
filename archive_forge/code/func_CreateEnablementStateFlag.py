from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.scc.manage import constants
def CreateEnablementStateFlag(module_type: constants.CustomModuleType, required: bool):
    """Creates an enablement state flag."""
    if module_type == constants.CustomModuleType.SHA:
        module_name = 'Security Health Analytics'
    elif module_type == constants.CustomModuleType.ETD:
        module_name = 'Event Threat Detection'
    return base.Argument('--enablement-state', required=required, default=None, help='Sets the enablement state of the {} custom module. Valid options are ENABLED, DISABLED, OR INHERITED.'.format(module_name))