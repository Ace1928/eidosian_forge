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
def _bgp_address_pools(self, args: parser_extensions.Namespace):
    """Constructs repeated proto message BareMetalBgpLbConfig.BareMetalLoadBalancerAddressPool."""
    if 'bgp_address_pools' not in args.GetSpecifiedArgsDict():
        return []
    address_pools = []
    for address_pool in args.bgp_address_pools:
        address_pools.append(messages.BareMetalLoadBalancerAddressPool(addresses=address_pool.get('addresses', []), avoidBuggyIps=address_pool.get('avoid-buggy-ips', None), manualAssign=address_pool.get('manual-assign', None), pool=address_pool.get('pool', None)))
    return address_pools