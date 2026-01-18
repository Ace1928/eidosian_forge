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
def _bgp_peer_configs(self, args: parser_extensions.Namespace):
    """Constructs repeated proto message BareMetalBgpPeerConfig."""
    if 'bgp_peer_configs' not in args.GetSpecifiedArgsDict():
        return []
    ret = []
    for peer_config in self.GetFlag(args, 'bgp_peer_configs'):
        msg = messages.BareMetalBgpPeerConfig(asn=peer_config.get('asn', None), controlPlaneNodes=peer_config.get('control-plane-nodes', []), ipAddress=peer_config.get('ip', None))
        ret.append(msg)
    return ret