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
def AddEnableConfidentialNodesFlag(parser, for_node_pool=False, hidden=False, is_update=False):
    """Adds a --enable-confidential-nodes flag to the given parser."""
    target = 'node pool' if for_node_pool else 'cluster'
    help_text = 'Enable confidential nodes for the {}. Enabling Confidential Nodes\nwill create nodes using Confidential VM\nhttps://cloud.google.com/compute/confidential-vm/docs/about-cvm.'.format(target)
    if is_update:
        help_text = '    Recreate all the nodes in the node pool to be confidential VM\n    https://cloud.google.com/compute/confidential-vm/docs/about-cvm.'
    parser.add_argument('--enable-confidential-nodes', help=help_text, default=None, hidden=hidden, action='store_true')