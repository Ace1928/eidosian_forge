from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def AddArchitectureFlag(parser, messages):
    architecture_enum_type = messages.Disk.ArchitectureValueValuesEnum
    excluded_enums = [architecture_enum_type.ARCHITECTURE_UNSPECIFIED.name]
    architecture_choices = sorted([e for e in architecture_enum_type.names() if e not in excluded_enums])
    return parser.add_argument('--architecture', choices=architecture_choices, help='Specifies the architecture or processor type that this disk can support. For available processor types on Compute Engine, see https://cloud.google.com/compute/docs/cpu-platforms.')