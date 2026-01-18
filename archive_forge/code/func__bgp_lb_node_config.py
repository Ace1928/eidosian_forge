from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _bgp_lb_node_config(self, bgp_lb_node_config):
    """Constructs proto message BareMetalNodeConfig."""
    node_ip = bgp_lb_node_config.get('nodeIP', '')
    if not node_ip:
        self._raise_bad_argument_exception_error('--bgp_lb_load_balancer_node_configs_from_file', 'nodeIP', 'BGP LB Node configs file')
    kwargs = {'nodeIp': node_ip, 'labels': self._node_labels(bgp_lb_node_config.get('labels', {}))}
    return messages.BareMetalNodeConfig(**kwargs)