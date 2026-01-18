from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def ReconcilingOnly():
    return base.Argument('--reconciling-only', action='store_true', help='If specified, only displays quota preferences in unresolved states. Default is false.')