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
def AddMinCpuPlatformFlag(parser, for_node_pool=False, hidden=False):
    """Adds the --min-cpu-platform flag to the parser.

  Args:
    parser: A given parser.
    for_node_pool: True if it's applied a non-default node pool.
    hidden: Whether or not to hide the help text.
  """
    if for_node_pool:
        help_text = 'When specified, the nodes for the new node pool will be scheduled on host with\nspecified CPU architecture or a newer one.\n\nExamples:\n\n  $ {command} node-pool-1 --cluster=example-cluster --min-cpu-platform=PLATFORM\n\n'
    else:
        help_text = "When specified, the nodes for the new cluster's default node pool will be\nscheduled on host with specified CPU architecture or a newer one.\n\nExamples:\n\n  $ {command} example-cluster --min-cpu-platform=PLATFORM\n\n"
    help_text += 'To list available CPU platforms in given zone, run:\n\n  $ gcloud beta compute zones describe ZONE --format="value(availableCpuPlatforms)"\n\nCPU platform selection is available only in selected zones.\n'
    parser.add_argument('--min-cpu-platform', metavar='PLATFORM', hidden=hidden, help=help_text)