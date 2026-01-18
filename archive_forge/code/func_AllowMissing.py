from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def AllowMissing():
    return base.Argument('--allow-missing', action='store_true', help='If specified and the quota preference is not found, a new one will be created. Default is false.')