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
def AddSpotFlag(parser, for_node_pool=False, hidden=False):
    """Adds a --spot flag to parser."""
    if for_node_pool:
        help_text = 'Create nodes using spot VM instances in the new node pool.\n\n  $ {command} node-pool-1 --cluster=example-cluster --spot\n'
    else:
        help_text = 'Create nodes using spot VM instances in the new cluster.\n\n  $ {command} example-cluster --spot\n'
    help_text += '\nNew nodes, including ones created by resize or recreate, will use spot\nVM instances.'
    parser.add_argument('--spot', action='store_true', help=help_text, hidden=hidden)