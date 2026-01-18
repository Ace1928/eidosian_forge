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
def AddNodePoolEnablePrivateNodes(parser):
    """Adds a --enable-private-nodes to the given node-pool parser."""
    help_text = '  Enables provisioning nodes with private IP addresses only.\n\n  The control plane still communicates with all nodes through\n  private IP addresses only, regardless of whether private\n  nodes are enabled or disabled.\n'
    parser.add_argument('--enable-private-nodes', default=None, action='store_true', help=help_text)