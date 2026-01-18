from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def TargetVpnGatewayArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='Target VPN Gateway', completer=TargetVpnGatewaysCompleter, plural=plural, required=required, regional_collection='compute.targetVpnGateways', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)