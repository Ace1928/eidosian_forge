from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def VpnTunnelArgumentForRoute(required=True):
    return compute_flags.ResourceArgument(resource_name='vpn tunnel', name='--next-hop-vpn-tunnel', completer=VpnTunnelsCompleter, plural=False, required=required, regional_collection='compute.vpnTunnels', short_help='The target VPN tunnel that will receive forwarded traffic.', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)