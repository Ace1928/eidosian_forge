from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
def PreferrenceId(positional=True, text='ID of the Quota Preference object, must be unique under its parent.'):
    if positional:
        return base.Argument('PREFERENCE_ID', type=str, help=text)
    else:
        return base.Argument('--preference-id', type=str, required=False, help=text)