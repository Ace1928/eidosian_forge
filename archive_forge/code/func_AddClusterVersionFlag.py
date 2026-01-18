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
def AddClusterVersionFlag(parser, suppressed=False, help=None):
    """Adds a --cluster-version flag to the given parser."""
    if help is None:
        help = 'The Kubernetes version to use for the master and nodes. Defaults to\nserver-specified.\n\nThe default Kubernetes version is available using the following command.\n\n  $ gcloud container get-server-config\n'
    return parser.add_argument('--cluster-version', help=help, hidden=suppressed)