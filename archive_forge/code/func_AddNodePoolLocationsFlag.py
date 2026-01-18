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
def AddNodePoolLocationsFlag(parser, for_create=False):
    """Adds a --node-locations flag for node pool to parser."""
    if for_create:
        help_text = "\nThe set of zones in which the node pool's nodes should be located.\n\nMultiple locations can be specified, separated by commas. For example:\n\n  $ {command} node-pool-1 --cluster=sample-cluster --node-locations=us-central1-a,us-central1-b"
    else:
        help_text = "Set of zones in which the node pool's nodes should be located.\nChanging the locations for a node pool will result in nodes being either created or removed\nfrom the node pool, depending on whether locations are being added or removed.\n\nMultiple locations can be specified, separated by commas. For example:\n\n  $ {command} node-pool-1 --cluster=sample-cluster --node-locations=us-central1-a,us-central1-b"
    parser.add_argument('--node-locations', type=arg_parsers.ArgList(min_length=1), metavar='ZONE', help=help_text)