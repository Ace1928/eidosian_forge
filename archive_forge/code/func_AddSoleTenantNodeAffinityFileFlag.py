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
def AddSoleTenantNodeAffinityFileFlag(parser, hidden=False):
    """Adds --sole-tenant-node-affinity-file flag to the given parser.

  Args:
    parser: A given parser.
    hidden: Indicates that the flags are hidden.
  """
    parser.add_argument('--sole-tenant-node-affinity-file', type=arg_parsers.YAMLFileContents(), hidden=hidden, help="      JSON/YAML file containing the configuration of desired sole tenant\n      nodes onto which this node pool could be backed by. These rules filter the\n      nodes according to their node affinity labels. A node's affinity labels\n      come from the node template of the group the node is in.\n\n      The file should contain a list of a JSON/YAML objects. For an example,\n      see https://cloud.google.com/compute/docs/nodes/provisioning-sole-tenant-vms#configure_node_affinity_labels.\n      The following list describes the fields:\n\n      *key*::: Corresponds to the node affinity label keys of\n      the Node resource.\n      *operator*::: Specifies the node selection type. Must be one of:\n        `IN`: Requires Compute Engine to seek for matched nodes.\n        `NOT_IN`: Requires Compute Engine to avoid certain nodes.\n      *values*::: Optional. A list of values which correspond to the node\n      affinity label values of the Node resource.\n      ")