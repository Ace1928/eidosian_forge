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
def _bgp_peer_configs_from_file(self, args: parser_extensions.Namespace):
    """Constructs proto message field address_pools."""
    if not args.bgp_lb_peer_configs_from_file:
        return []
    peer_configs = args.bgp_lb_peer_configs_from_file.get('bgpPeerConfigs', [])
    if not peer_configs:
        self._raise_bad_argument_exception_error('--bgp_lb_peer_configs_from_file', 'bgpPeerConfigs', 'BGP LB peer configs file')
    peer_configs_messages = []
    for peer_config in peer_configs:
        peer_configs_messages.append(self._peer_configs(peer_config))
    return peer_configs_messages