from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
def ExternalVpnGatewayArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='external VPN gateway', completer=ExternalVpnGatewaysCompleter, plural=plural, required=required, global_collection='compute.externalVpnGateways')