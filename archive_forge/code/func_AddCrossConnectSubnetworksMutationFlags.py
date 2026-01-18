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
def AddCrossConnectSubnetworksMutationFlags(parser, hidden=True):
    """Adds flags for mutating cross connect subnetworks in cluster update."""
    add_help_text = ' '
    remove_help_text = ' '
    clear_help_text = ' '
    parser.add_argument('--add-cross-connect-subnetworks', help=add_help_text, hidden=hidden, type=arg_parsers.ArgList(min_length=1), metavar='SUBNETS')
    parser.add_argument('--remove-cross-connect-subnetworks', help=remove_help_text, hidden=hidden, type=arg_parsers.ArgList(min_length=1), metavar='SUBNETS')
    parser.add_argument('--clear-cross-connect-subnetworks', help=clear_help_text, hidden=hidden, default=None, action='store_true')