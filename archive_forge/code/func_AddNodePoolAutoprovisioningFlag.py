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
def AddNodePoolAutoprovisioningFlag(parser, hidden=True):
    """Adds --enable-autoprovisioning flag for node pool to parser.

  Args:
    parser: A given parser.
    hidden: If true, suppress help text for added options.
  """
    parser.add_argument('--enable-autoprovisioning', help="Enables Cluster Autoscaler to treat the node pool as if it was autoprovisioned.\n\nCluster Autoscaler will be able to delete the node pool if it's unneeded.", hidden=hidden, default=None, action='store_true')