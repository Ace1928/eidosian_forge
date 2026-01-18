from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddNodePoolConfig(parser: parser_arguments.ArgumentInterceptor, is_update=False):
    """Adds a command group to set the node pool config.

  Args:
    parser: The argparse parser to add the flag to.
    is_update: bool, whether the flag is for update command or not.
  """
    required = not is_update
    bare_metal_node_pool_config_group = parser.add_group(required=required, help='Anthos on bare metal node pool configuration.')
    _AddNodeConfigs(bare_metal_node_pool_config_group, is_update)
    _AddNodeLabels(bare_metal_node_pool_config_group)
    _AddNodeTaints(bare_metal_node_pool_config_group)
    _AddBareMetalKubeletConfig(bare_metal_node_pool_config_group, is_update)