from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddAutoscaleRolloutPolicyFlag(parser, for_node_pool=True, hidden=True):
    """Adds --autoscaled-rollout-policy flag to the parser."""
    autoscaled_rollout_policy_help = 'Autoscaled rollout policy options for blue-green upgrade.\n'
    if for_node_pool:
        autoscaled_rollout_policy_help += '  $ {command} node-pool-1 --cluster=example-cluster  --autoscaled-rollout-policy\n'
    parser.add_argument('--autoscaled-rollout-policy', help=autoscaled_rollout_policy_help, hidden=hidden, action='store_true')