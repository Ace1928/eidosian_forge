from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def Dimensions():
    return base.Argument('--dimensions', type=arg_parsers.ArgDict(), metavar='KEY=VALUE', action=arg_parsers.UpdateAction, help='Dimensions of the quota.')