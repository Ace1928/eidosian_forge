from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.netapp import flags
from googlecloudsdk.command_lib.netapp import util as netapp_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddVolumeSecurityStyleArg(parser, messages):
    """Adds the --security-style arg to the arg parser."""
    security_style_arg = arg_utils.ChoiceEnumMapper('--security-style', messages.Volume.SecurityStyleValueValuesEnum, help_str='The security style of the Volume. This can either be\n          UNIX or NTFS.\n        ', custom_mappings={'UNIX': ('unix', 'UNIX security style for Volume'), 'NTFS': ('ntfs', 'NTFS security style for Volume.')}, default='SECURITY_STYLE_UNSPECIFIED')
    security_style_arg.choice_arg.AddToParser(parser)