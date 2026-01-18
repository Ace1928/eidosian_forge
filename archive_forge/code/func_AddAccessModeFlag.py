from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def AddAccessModeFlag(parser, messages):
    if hasattr(messages.Disk, 'AccessModeValueValuesEnum'):
        access_mode_enum_type = messages.Disk.AccessModeValueValuesEnum
        return parser.add_argument('--access-mode', choices=access_mode_enum_type.names(), help='Specifies the access mode that the disk can support.')