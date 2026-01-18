from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.scc.manage import constants
def CreateModuleTypeFlag(required=True) -> base.Argument:
    return base.Argument('--module-type', required=required, metavar='MODULE_TYPE', help='Type of the custom module. For a list of valid module types please visit https://cloud.google.com/security-command-center/docs/custom-modules-etd-overview#custom_modules_and_templates.')