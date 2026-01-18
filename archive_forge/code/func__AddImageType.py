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
def _AddImageType(vmware_node_config_group, for_update=False):
    """Adds a flag to specify the node pool image type.

  Args:
    vmware_node_config_group: The argparse parser to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    required = False if for_update else True
    vmware_node_config_group.add_argument('--image-type', required=required, help='OS image type to use on node pool instances.')