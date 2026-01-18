from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddIslandModeCIDRConfig(bare_metal_network_config_group, is_update=False):
    """Adds island mode CIDR config related flags.

  Args:
    bare_metal_network_config_group: The parent group to add the flag to.
    is_update: bool, whether the flag is for update command or not.
  """
    bare_metal_island_mode_cidr_config_group = bare_metal_network_config_group.add_group(help='Island mode CIDR network configuration.')
    _AddServiceAddressCIDRBlocks(bare_metal_island_mode_cidr_config_group, is_update)