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
def _bgp_load_balancer_node_configs(self, args: parser_extensions.Namespace):
    """Constructs repeated proto message BareMetalBgpLbConfig.BareMetalLoadBalancerNodePoolConfig.BareMetalNodePoolConfig.BareMetalNodeConfig."""
    if 'bgp_load_balancer_node_configs' not in args.GetSpecifiedArgsDict():
        return []
    node_configs = []
    for node_config in self.GetFlag(args, 'bgp_load_balancer_node_configs'):
        node_configs.append(self.node_config(node_config))
    return node_configs