from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def ValidateOnly():
    return base.Argument('--validate-only', action='store_true', help='If specified, only validates the request, but does not actually update. Note that a request being valid does not mean that the request is guaranteed to be fulfilled. Default is false.')