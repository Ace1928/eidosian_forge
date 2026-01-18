from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def PreferredValue():
    return base.Argument('--preferred-value', required=True, help='Preferred value. Must be greater than or equal to -1. If set to -1, it means the value is "unlimited".')