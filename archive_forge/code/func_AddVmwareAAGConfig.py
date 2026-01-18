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
def AddVmwareAAGConfig(parser: parser_arguments.ArgumentInterceptor, for_update=False):
    """Adds flags to specify VMware user cluster node pool anti-affinity group configurations.

  Args:
    parser: The argparse parser to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    vmware_aag_config_group = parser.add_group(help='Anti-affinity group configurations')
    if for_update:
        enable_aag_config_mutex_group = vmware_aag_config_group.add_group(mutex=True)
        surface = enable_aag_config_mutex_group
    else:
        surface = vmware_aag_config_group
    surface.add_argument('--disable-aag-config', action='store_true', help='If set, spread nodes across at least three physical hosts (requires at least three hosts). Enabled by default.')
    if for_update:
        surface.add_argument('--enable-aag-config', action='store_true', help='If set, enable anti-affinity group config.')