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
def _AddVIPConfig(bare_metal_load_balancer_config_group, is_update=False):
    """Adds flags to set VIPs used by the load balancer.

  Args:
    bare_metal_load_balancer_config_group: The parent group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    if is_update:
        return None
    bare_metal_vip_config_group = bare_metal_load_balancer_config_group.add_group(help=' VIPs used by the load balancer.', required=True)
    bare_metal_vip_config_group.add_argument('--control-plane-vip', required=True, help='VIP for the Kubernetes API of this cluster.')
    bare_metal_vip_config_group.add_argument('--ingress-vip', required=True, help='VIP for ingress traffic into this cluster.')