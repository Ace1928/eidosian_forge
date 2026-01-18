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
def AddNodePoolSoakDurationFlag(parser, for_node_pool=False, hidden=False):
    """Adds --node-pool-soak-duration flag to the parser."""
    node_pool_soak_duration_help = 'Time in seconds to be spent waiting during blue-green upgrade before\ndeleting the blue pool and completing the upgrade.\n\n'
    if for_node_pool:
        node_pool_soak_duration_help += '  $ {command} node-pool-1 --cluster=example-cluster  --node-pool-soak-duration=600s\n'
    else:
        node_pool_soak_duration_help += '  $ {command} example-cluster  --node-pool-soak-duration=600s\n'
    parser.add_argument('--node-pool-soak-duration', type=str, help=node_pool_soak_duration_help, hidden=hidden)