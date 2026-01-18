from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
def ExternalVpnGatewayArgumentForVpnTunnel(required=False):
    return compute_flags.ResourceArgument(name='--peer-external-gateway', resource_name='external VPN gateway', completer=ExternalVpnGatewaysCompleter, required=required, short_help='Peer side external VPN gateway representing the remote tunnel endpoint, this flag is used when creating HA VPN tunnels from Google Cloud to your external VPN gateway.Either --peer-external-gateway or --peer-gcp-gateway must be specified when creating VPN tunnels from High Available VPN gateway.', global_collection='compute.externalVpnGateways')