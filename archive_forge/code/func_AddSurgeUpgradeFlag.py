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
def AddSurgeUpgradeFlag(parser, for_node_pool=False, default=None):
    """Adds --max-surge-upgrade flag to the parser."""
    if for_node_pool:
        max_surge_help = "Number of extra (surge) nodes to be created on each upgrade of the node pool.\n\nSpecifies the number of extra (surge) nodes to be created during this node\npool's upgrades. For example, running the following command will result in\ncreating an extra node each time the node pool is upgraded:\n\n  $ {command} node-pool-1 --cluster=example-cluster --max-surge-upgrade=1   --max-unavailable-upgrade=0\n\nMust be used in conjunction with '--max-unavailable-upgrade'.\n"
    else:
        max_surge_help = "Number of extra (surge) nodes to be created on each upgrade of a node pool.\n\nSpecifies the number of extra (surge) nodes to be created during this node\npool's upgrades. For example, running the following command will result in\ncreating an extra node each time the node pool is upgraded:\n\n  $ {command} example-cluster --max-surge-upgrade=1 --max-unavailable-upgrade=0\n\nMust be used in conjunction with '--max-unavailable-upgrade'.\n"
    parser.add_argument('--max-surge-upgrade', type=int, default=default, help=max_surge_help, hidden=False)