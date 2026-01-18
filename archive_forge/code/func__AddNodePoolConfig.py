from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddNodePoolConfig(bare_metal_control_plane_node_pool_config_group, is_update=False):
    """Adds a command group to set the node pool config.

  Args:
    bare_metal_control_plane_node_pool_config_group: The argparse parser to add
      the flag to.
    is_update: bool, whether the flag is for update command or not.
  """
    required = not is_update
    bare_metal_node_pool_config_group = bare_metal_control_plane_node_pool_config_group.add_group(help='Anthos on bare metal node pool configuration for control plane nodes.', required=required)
    bare_metal_node_config_group = bare_metal_node_pool_config_group.add_group(help='Anthos on bare metal node configuration for control plane nodes.', required=required)
    _AddControlPlaneNodeConfigs(bare_metal_node_config_group, is_update)
    _AddControlPlaneNodeLabels(bare_metal_node_config_group)
    _AddControlPlaneNodeTaints(bare_metal_node_config_group)