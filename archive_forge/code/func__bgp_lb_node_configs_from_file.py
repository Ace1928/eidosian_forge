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
def _bgp_lb_node_configs_from_file(self, args: parser_extensions.Namespace):
    """Constructs proto message field node_configs."""
    if not args.bgp_lb_load_balancer_node_configs_from_file:
        return []
    bgp_lb_node_configs = args.bgp_lb_load_balancer_node_configs_from_file.get('nodeConfigs', [])
    if not bgp_lb_node_configs:
        self._raise_bad_argument_exception_error('--bgp_lb_load_balancer_node_configs_from_file', 'nodeConfigs', 'BGP LB Node configs file')
    bgp_lb_node_configs_messages = []
    for bgp_lb_node_config in bgp_lb_node_configs:
        bgp_lb_node_configs_messages.append(self._bgp_lb_node_config(bgp_lb_node_config))
    return bgp_lb_node_configs_messages