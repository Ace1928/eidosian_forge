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
def _bgp_address_pools_from_file(self, args: parser_extensions.Namespace):
    """Constructs proto message field address_pools."""
    if not args.bgp_lb_address_pools_from_file:
        return []
    address_pools = args.bgp_lb_address_pools_from_file.get('addressPools', [])
    if not address_pools:
        self._raise_bad_argument_exception_error('--bgp_lb_address_pools_from_file', 'addressPools', 'BGP LB address pools file')
    address_pool_messages = []
    for address_pool in address_pools:
        address_pool_messages.append(self._bgp_lb_address_pool(address_pool))
    return address_pool_messages