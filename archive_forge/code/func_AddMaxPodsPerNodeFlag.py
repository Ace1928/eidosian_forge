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
def AddMaxPodsPerNodeFlag(parser, for_node_pool=False, hidden=False):
    """Adds max pod number constraints flags to the parser.

  Args:
    parser: A given parser.
    for_node_pool: True if it's applied to a node pool. False if it's applied to
      a cluster.
    hidden: Whether or not to hide the help text.
  """
    parser.add_argument('--max-pods-per-node', default=None, help="The max number of pods per node for this node pool.\n\nThis flag sets the maximum number of pods that can be run at the same time on a\nnode. This will override the value given with --default-max-pods-per-node flag\nset at the cluster level.\n\nMust be used in conjunction with '--enable-ip-alias'.\n", hidden=hidden, type=int)
    if not for_node_pool:
        parser.add_argument('--default-max-pods-per-node', default=None, help="The default max number of pods per node for node pools in the cluster.\n\nThis flag sets the default max-pods-per-node for node pools in the cluster. If\n--max-pods-per-node is not specified explicitly for a node pool, this flag\nvalue will be used.\n\nMust be used in conjunction with '--enable-ip-alias'.\n", hidden=hidden, type=int)