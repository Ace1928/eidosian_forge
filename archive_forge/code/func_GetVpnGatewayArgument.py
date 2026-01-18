from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetVpnGatewayArgument(required=True, plural=False):
    """Returns the resource argument object for the VPN gateway flag."""
    return compute_flags.ResourceArgument(resource_name='VPN Gateway', completer=VpnGatewaysCompleter, plural=plural, custom_plural='VPN Gateways', required=required, regional_collection='compute.vpnGateways', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)