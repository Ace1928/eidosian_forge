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
def AddNetworkPerformanceConfigFlags(parser, hidden=True):
    """Adds config flags for advanced networking bandwidth tiers."""
    network_perf_config_help = '      Configures network performance settings for the node pool.\n      If this flag is not specified, the pool will be created\n      with its default network performance configuration.\n\n      *total-egress-bandwidth-tier*::: Total egress bandwidth is the available\n      outbound bandwidth from a VM, regardless of whether the traffic\n      is going to internal IP or external IP destinations.\n      The following tier values are allowed: [{tier_values}]\n\n      '.format(tier_values=','.join(['TIER_UNSPECIFIED', 'TIER_1']))
    spec = {'total-egress-bandwidth-tier': str}
    parser.add_argument('--network-performance-configs', type=arg_parsers.ArgDict(spec=spec), action='append', metavar='PROPERTY=VALUE', hidden=hidden, help=network_perf_config_help)