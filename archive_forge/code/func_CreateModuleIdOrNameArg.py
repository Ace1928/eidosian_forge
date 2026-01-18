from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.scc.manage import constants
def CreateModuleIdOrNameArg(module_type: constants.CustomModuleType) -> base.Argument:
    """A positional argument representing a custom module ID or name."""
    return base.Argument('module_id_or_name', help='The custom module ID or name. The expected format is {parent}/[locations/global]/MODULE_TYPE/{module_id} or just {module_id}. Where module_id is a numeric identifier 1-20 characters\n      in length. Parent is of the form organizations/{id}, projects/{id or name},\n      folders/{id}.'.replace('MODULE_TYPE', module_type))