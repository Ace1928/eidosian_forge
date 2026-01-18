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
def AddStandardRolloutPolicyFlag(parser, for_node_pool=False, hidden=False):
    """Adds --standard-rollout-policy flag to the parser."""
    standard_rollout_policy_help = 'Standard rollout policy options for blue-green upgrade.\n\nBatch sizes are specified by one of, batch-node-count or batch-percent.\nThe duration between batches is specified by batch-soak-duration.\n\n'
    if for_node_pool:
        standard_rollout_policy_help += '  $ {command} node-pool-1 --cluster=example-cluster  --standard-rollout-policy=batch-node-count=3,batch-soak-duration=60s\n\n  $ {command} node-pool-1 --cluster=example-cluster  --standard-rollout-policy=batch-percent=0.3,batch-soak-duration=60s\n'
    else:
        standard_rollout_policy_help += '  $ {command} example-cluster  --standard-rollout-policy=batch-node-count=3,batch-soak-duration=60s\n\n  $ {command} example-cluster  --standard-rollout-policy=batch-percent=0.3,batch-soak-duration=60s\n'
    spec = {'batch-node-count': int, 'batch-percent': float, 'batch-soak-duration': str}
    parser.add_argument('--standard-rollout-policy', help=standard_rollout_policy_help, hidden=hidden, metavar='batch-node-count=BATCH_NODE_COUNT,batch-percent=BATCH_NODE_PERCENTAGE,batch-soak-duration=BATCH_SOAK_DURATION', type=arg_parsers.ArgDict(spec=spec))