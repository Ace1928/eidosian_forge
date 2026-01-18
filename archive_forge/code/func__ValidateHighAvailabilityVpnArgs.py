from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.vpn_tunnels import vpn_tunnels_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.external_vpn_gateways import (
from googlecloudsdk.command_lib.compute.routers import flags as router_flags
from googlecloudsdk.command_lib.compute.target_vpn_gateways import (
from googlecloudsdk.command_lib.compute.vpn_gateways import (flags as
from googlecloudsdk.command_lib.compute.vpn_tunnels import flags
def _ValidateHighAvailabilityVpnArgs(self, args):
    if args.IsSpecified('vpn_gateway'):
        if not args.IsSpecified('interface'):
            raise exceptions.InvalidArgumentException('--interface', 'When creating Highly Available VPN tunnels, the VPN gateway interface must be specified using the --interface flag.')
        if not args.IsSpecified('router'):
            raise exceptions.InvalidArgumentException('--router', 'When creating Highly Available VPN tunnels, a Cloud Router must be specified using the --router flag.')
        if not args.IsSpecified('peer_gcp_gateway') and (not args.IsSpecified('peer_external_gateway')):
            raise exceptions.InvalidArgumentException('--peer-gcp-gateway', 'When creating Highly Available VPN tunnels, either --peer-gcp-gateway or --peer-external-gateway must be specified.')
        if args.IsSpecified('peer_external_gateway') and (not args.IsSpecified('peer_external_gateway_interface')):
            raise exceptions.InvalidArgumentException('--peer-external-gateway-interface', 'The flag --peer-external-gateway-interface must be specified along with --peer-external-gateway.')
        if args.IsSpecified('local_traffic_selector'):
            raise exceptions.InvalidArgumentException('--local-traffic-selector', 'Cannot specify local traffic selector with Highly Available VPN tunnels.')
        if args.IsSpecified('remote_traffic_selector'):
            raise exceptions.InvalidArgumentException('--remote-traffic-selector', 'Cannot specify remote traffic selector with Highly Available VPN tunnels.')
        if args.IsSpecified('peer_address'):
            raise exceptions.InvalidArgumentException('--peer-address', 'Cannot specify the flag peer address with Highly Available VPN tunnels.')