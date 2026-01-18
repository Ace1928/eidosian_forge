from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetPeerVpnGatewayArgumentForOtherResource(required=False):
    """Returns the flag for specifying the peer VPN gateway."""
    return compute_flags.ResourceArgument(name='--peer-gcp-gateway', resource_name='VPN Gateway', completer=VpnGatewaysCompleter, plural=False, required=required, regional_collection='compute.vpnGateways', short_help='Peer side Highly Available VPN gateway representing the remote tunnel endpoint, this flag is used when creating HA VPN tunnels from Google Cloud to Google Cloud.Either --peer-external-gateway or --peer-gcp-gateway must be specified when creating VPN tunnels from High Available VPN gateway.', region_explanation='Should be the same as region, if not specified, it will be automatically set.', detailed_help='        Reference to the peer side Highly Available VPN gateway.\n        ')