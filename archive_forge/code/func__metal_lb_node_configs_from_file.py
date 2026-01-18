from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _metal_lb_node_configs_from_file(self, args: parser_extensions.Namespace):
    """Constructs proto message field node_configs."""
    if not args.metal_lb_load_balancer_node_configs_from_file:
        return []
    metal_lb_node_configs = args.metal_lb_load_balancer_node_configs_from_file.get('nodeConfigs', [])
    if not metal_lb_node_configs:
        raise exceptions.BadArgumentException('--metal_lb_load_balancer_node_configs_from_file', 'Missing field [nodeConfigs] in Metal LB Node configs file.')
    metal_lb_node_configs_messages = []
    for metal_lb_node_config in metal_lb_node_configs:
        metal_lb_node_configs_messages.append(self._metal_lb_node_config(metal_lb_node_config))
    return metal_lb_node_configs_messages