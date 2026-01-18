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
def AddBestEffortProvisionFlags(parser, hidden=False):
    """Adds the argument to enable best effort provisioning."""
    group_text = '      Specifies minimum number of nodes to be created when best effort\n      provisioning enabled.\n  '
    enable_best_provision = '      Enable best effort provision for nodes\n  '
    min_provision_nodes = '      Specifies the minimum number of nodes to be provisioned during creation\n  '
    group = parser.add_group(help=group_text, hidden=hidden)
    group.add_argument('--enable-best-effort-provision', default=None, help=enable_best_provision, action='store_true')
    group.add_argument('--min-provision-nodes', default=None, type=int, help=min_provision_nodes)