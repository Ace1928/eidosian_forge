from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddAutoscalingPolicyArgToParser(parser, required_mode=False):
    """Add autoscaling configuration  arguments to parser."""
    group = parser.add_group(help='Autoscaling policy for node groups.')
    group.add_argument('--autoscaler-mode', required=required_mode, choices={'on': 'to permit autoscaling to scale in and out.', 'only-scale-out': 'to permit autoscaling to scale only out and not in.', 'off': 'to turn off autoscaling.'}, help='Set the mode of an autoscaler for a node group.')
    group.add_argument('--min-nodes', type=int, help='\nThe minimum size of the node group. Default is 0 and must be an integer value\nsmaller than or equal to `--max-nodes`.\n')
    group.add_argument('--max-nodes', type=int, help="\nThe maximum size of the node group. Must be smaller or equal to 100 and larger\nthan or equal to `--min-nodes`. Must be specified if `--autoscaler-mode` is not\n``off''.\n")