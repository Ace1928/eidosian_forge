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
def _AddManualLbConfig(lb_config_mutex_group, for_update=False):
    """Adds flags for Manual load balancer.

  Args:
    lb_config_mutex_group: The parent mutex group to add the flags to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    if for_update:
        return
    manual_lb_config_group = lb_config_mutex_group.add_group(help=textwrap.dedent('        Manual load balancer configuration.\n\n        With manual load balancing mode, DHCP is not supported. Specify static IP addresses for cluster nodes instead.\n        For more details, see https://cloud.google.com/anthos/clusters/docs/on-prem/latest/how-to/manual-load-balance#setting_aside_node_ip_addresses.\n        '))
    manual_lb_config_group.add_argument('--ingress-http-node-port', help="NodePort for ingress service's http.", type=int)
    manual_lb_config_group.add_argument('--ingress-https-node-port', help="NodePort for ingress service's https.", type=int)
    manual_lb_config_group.add_argument('--control-plane-node-port', help='NodePort for control plane service.', type=int)
    manual_lb_config_group.add_argument('--konnectivity-server-node-port', help='NodePort for konnectivity service running as a sidecar in each kube-apiserver pod.', type=int)