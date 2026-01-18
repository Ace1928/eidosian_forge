from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def AllowsQuotaDecreaseBelowUsage():
    return base.Argument('--allow-quota-decrease-below-usage', action='store_true', help='If specified, allows consumers to reduce their effective limit below their quota usage. Default is false.')