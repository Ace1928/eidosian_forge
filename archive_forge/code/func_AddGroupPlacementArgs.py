from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddGroupPlacementArgs(parser, messages, track):
    """Adds flags specific to group placement resource policies."""
    parser.add_argument('--vm-count', type=arg_parsers.BoundedInt(lower_bound=1), help='Number of instances targeted by the group placement policy. Google does not recommend that you use this flag unless you use a compact policy and you want your policy to work only if it contains this exact number of VMs.')
    parser.add_argument('--availability-domain-count', type=arg_parsers.BoundedInt(lower_bound=1), help='Number of availability domain in the group placement policy.')
    GetCollocationFlagMapper(messages, track).choice_arg.AddToParser(parser)
    if track == base.ReleaseTrack.ALPHA:
        GetAvailabilityDomainScopeFlagMapper(messages).choice_arg.AddToParser(parser)
        parser.add_argument('--tpu-topology', type=str, help='Specifies the shape of the TPU pod slice.')
    if track in (base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA):
        parser.add_argument('--max-distance', type=arg_parsers.BoundedInt(lower_bound=1, upper_bound=3), help='Specifies the number of max logical switches between VMs.')