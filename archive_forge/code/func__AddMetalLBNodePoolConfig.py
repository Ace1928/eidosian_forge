from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddMetalLBNodePoolConfig(metal_lb_config_group):
    """Adds a command group to set the node pool config for MetalLB load balancer.

  Args:
   metal_lb_config_group: The argparse parser to add the flag to.
  """
    bare_metal_metal_lb_node_pool_config_group = metal_lb_config_group.add_group(help='Anthos on bare metal node pool configuration for MetalLB load balancer nodes.')
    bare_metal_metal_lb_node_config = bare_metal_metal_lb_node_pool_config_group.add_group(help='MetalLB Node Pool configuration.')
    _AddMetalLBNodeConfigs(bare_metal_metal_lb_node_config)
    _AddMetalLBNodeLabels(bare_metal_metal_lb_node_config)
    _AddMetalLBNodeTaints(bare_metal_metal_lb_node_config)