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
def _GetPeerExternalGateway(self, api_resource_registry, args):
    if args.IsSpecified('peer_external_gateway'):
        peer_external_gateway = self._EXTERNAL_VPN_GATEWAY_ARG.ResolveAsResource(args, api_resource_registry).SelfLink()
        return peer_external_gateway
    return None