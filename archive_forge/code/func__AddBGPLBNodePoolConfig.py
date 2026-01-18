from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddBGPLBNodePoolConfig(bgp_lb_config_group):
    """Adds a command group to set the node pool config for BGP LB load balancer.

  Args:
   bgp_lb_config_group: The argparse parser to add the flag to.
  """
    bare_metal_bgp_lb_node_pool_config_group = bgp_lb_config_group.add_group(help='Anthos on bare metal node pool configuration for BGP LB load balancer nodes.')
    bare_metal_bgp_lb_node_config = bare_metal_bgp_lb_node_pool_config_group.add_group(help='BGP LB Node Pool configuration.')
    _AddBGPLBNodeConfigs(bare_metal_bgp_lb_node_config)
    _AddBGPLBNodeLabels(bare_metal_bgp_lb_node_config)
    _AddBGPLBNodeTaints(bare_metal_bgp_lb_node_config)