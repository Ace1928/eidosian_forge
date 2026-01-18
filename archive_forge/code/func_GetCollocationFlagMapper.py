from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def GetCollocationFlagMapper(messages, track):
    """Gets collocation flag mapper for resource policies."""
    custom_mappings = {'UNSPECIFIED_COLLOCATION': ('unspecified-collocation', 'Unspecified network latency between VMs placed on the same availability domain. This is the default behavior.'), 'COLLOCATED': ('collocated', 'Low network latency between more VMs placed on the same availability domain.')}
    if track == base.ReleaseTrack.ALPHA:
        custom_mappings.update({'CLUSTERED': ('clustered', 'Lowest network latency between VMs placed on the same availability domain.')})
    return arg_utils.ChoiceEnumMapper('--collocation', messages.ResourcePolicyGroupPlacementPolicy.CollocationValueValuesEnum, custom_mappings=custom_mappings, default=None, help_str='Collocation specifies whether to place VMs inside the sameavailability domain on the same low-latency network.')