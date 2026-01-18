from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddServiceAddressCIDRBlocks(bare_metal_island_mode_cidr_config_group, is_update=False):
    """Adds a flag to specify the IPv4 address ranges used in the services in the cluster.

  Args:
    bare_metal_island_mode_cidr_config_group: The parent group to add the flag
      to.
    is_update: bool, whether the flag is for update command or not.
  """
    required = not is_update
    bare_metal_island_mode_cidr_config_group.add_argument('--island-mode-service-address-cidr-blocks', metavar='SERVICE_ADDRESS', required=required, type=arg_parsers.ArgList(min_length=1), help='IPv4 address range for all services in the cluster.')