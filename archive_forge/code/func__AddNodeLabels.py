from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddNodeLabels(vmware_node_config_group):
    """Adds a flag to specify the labels in the node pool.

  Args:
    vmware_node_config_group: The parent group to add the flag to.
  """
    vmware_node_config_group.add_argument('--node-labels', metavar='KEY=VALUE', help='Kubernetes labels (key/value pairs) to be applied to each node.', type=arg_parsers.ArgDict())