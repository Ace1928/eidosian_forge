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
def _AddBootDiskSizeGb(vmware_node_config_group):
    """Adds a flag to specify the boot disk size in GB in the node pool.

  Args:
    vmware_node_config_group: The parent group to add the flag to.
  """
    vmware_node_config_group.add_argument('--boot-disk-size', help='Size of VMware disk to be used during creation in GB.', type=arg_parsers.BinarySize(default_unit='GB', type_abbr='GB'))