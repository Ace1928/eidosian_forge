from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.scc.manage import constants
def CreateDisplayNameFlag(required=True) -> base.Argument:
    return base.Argument('--display-name', required=required, metavar='DISPLAY-NAME', help='The display name of the custom module.')