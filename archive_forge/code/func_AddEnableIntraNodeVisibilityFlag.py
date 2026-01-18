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
def AddEnableIntraNodeVisibilityFlag(parser, hidden=False):
    """Adds --enable-intra-node-visibility flag to the parser.

  When enabled, the intra-node traffic is visible to VPC network.

  Args:
    parser: A given parser.
    hidden: If true, suppress help text for added options.
  """
    parser.add_argument('--enable-intra-node-visibility', default=None, hidden=hidden, action='store_true', help='Enable Intra-node visibility for this cluster.\n\nEnabling intra-node visibility makes your intra-node pod-to-pod traffic\nvisible to the networking fabric. With this feature, you can use VPC flow\nlogging or other VPC features for intra-node traffic.\n\nEnabling it on an existing cluster causes the cluster\nmaster and the cluster nodes to restart, which might cause a disruption.\n')