from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddBGPLoadBalancerNodePoolConfig(bgp_lb_config_group, is_update=False):
    bgp_lb_node_pool_config_group = bgp_lb_config_group.add_group()
    bgp_node_pool_config_group = bgp_lb_node_pool_config_group.add_group()
    _AddBGPNodeConfigs(bgp_node_pool_config_group, is_update=is_update)
    _AddBGPNodeTaints(bgp_node_pool_config_group)
    _AddBGPNodeLabels(bgp_node_pool_config_group)
    _AddBGPKubeletConfig(bgp_node_pool_config_group, is_update=is_update)