from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def AllowHighPercentageQuotaDecrease():
    return base.Argument('--allow-high-percentage-quota-decrease', action='store_true', help='If specified, allows consumers to reduce their effective limit by more than 10 percent. Default is false.')