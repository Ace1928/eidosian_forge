from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
class ExternalVpnGatewaysCompleter(compute_completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(ExternalVpnGatewaysCompleter, self).__init__(collection='compute.externalVpnGateways', list_command='alpha compute external-vpn-gateways list --uri', **kwargs)