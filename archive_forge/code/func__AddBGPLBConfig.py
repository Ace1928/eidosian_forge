from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddBGPLBConfig(lb_config_mutex_group, is_update=False):
    """Adds flags for bgpLB load balancer.

  Args:
    lb_config_mutex_group: The parent mutex group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    bgp_lb_config_group = lb_config_mutex_group.add_group('BGP LB Configuration')
    _AddBGPLBAsn(bgp_lb_config_group, is_update)
    _AddBGPLBPeerConfigs(bgp_lb_config_group, is_update)
    _AddBGPLBAddressPools(bgp_lb_config_group, is_update)
    _AddBGPLBNodePoolConfig(bgp_lb_config_group)