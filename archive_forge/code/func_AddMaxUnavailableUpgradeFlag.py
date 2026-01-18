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
def AddMaxUnavailableUpgradeFlag(parser, for_node_pool=False, is_create=False):
    """Adds --max-unavailable-upgrade flag to the parser."""
    if for_node_pool:
        if is_create:
            max_unavailable_upgrade_help = "Number of nodes that can be unavailable at the same time on each upgrade of the\nnode pool.\n\nSpecifies the number of nodes that can be unavailable at the same time during\nthis node pool's upgrades. For example, running the following command will\nresult in having 3 nodes being upgraded in parallel (1 + 2), but keeping always\nat least 3 (5 - 2) available each time the node pool is upgraded:\n\n  $ {command} node-pool-1 --cluster=example-cluster --num-nodes=5   --max-surge-upgrade=1 --max-unavailable-upgrade=2\n\nMust be used in conjunction with '--max-surge-upgrade'.\n"
        else:
            max_unavailable_upgrade_help = "Number of nodes that can be unavailable at the same time on each upgrade of the\nnode pool.\n\nSpecifies the number of nodes that can be unavailable at the same time during\nthis node pool's upgrades. For example, assume the node pool has 5 nodes,\nrunning the following command will result in having 3 nodes being upgraded in\nparallel (1 + 2), but keeping always at least 3 (5 - 2) available each time the\nnode pool is upgraded:\n\n  $ {command} node-pool-1 --cluster=example-cluster --max-surge-upgrade=1   --max-unavailable-upgrade=2\n\nMust be used in conjunction with '--max-surge-upgrade'.\n"
    else:
        max_unavailable_upgrade_help = "Number of nodes that can be unavailable at the same time on each upgrade of a\nnode pool.\n\nSpecifies the number of nodes that can be unavailable at the same time while\nthis node pool is being upgraded. For example, running the following command\nwill result in having 3 nodes being upgraded in parallel (1 + 2), but keeping\nalways at least 3 (5 - 2) available each time the node pool is upgraded:\n\n   $ {command} example-cluster --num-nodes=5 --max-surge-upgrade=1      --max-unavailable-upgrade=2\n\nMust be used in conjunction with '--max-surge-upgrade'.\n"
    parser.add_argument('--max-unavailable-upgrade', type=int, default=None, help=max_unavailable_upgrade_help, hidden=False)