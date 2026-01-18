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
def _AddEnableLoadBalancer(vmware_node_config_group, for_update=False):
    """Adds a flag to enable load balancer in the node pool.

  Args:
    vmware_node_config_group: The parent group to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    if for_update:
        enable_lb_mutex_group = vmware_node_config_group.add_group(mutex=True)
        surface = enable_lb_mutex_group
    else:
        surface = vmware_node_config_group
    surface.add_argument('--enable-load-balancer', action='store_const', const=True, help='If set, enable the use of load balancer on the node pool instances.')
    if for_update:
        surface.add_argument('--disable-load-balancer', action='store_const', const=True, help='If set, disable the use of load balancer on the node pool instances.')