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
def _AddVmwareVipConfig(vmware_load_balancer_config_group, for_update=False):
    """Adds flags to set VIPs used by the load balancer..

  Args:
    vmware_load_balancer_config_group: The parent group to add the flags to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    required = False if for_update else True
    if for_update:
        return
    vmware_vip_config_group = vmware_load_balancer_config_group.add_group(help=' VIPs used by the load balancer', required=required)
    vmware_vip_config_group.add_argument('--control-plane-vip', required=required, help='VIP for the Kubernetes API of this cluster.')
    vmware_vip_config_group.add_argument('--ingress-vip', required=required, help='VIP for ingress traffic into this cluster.')