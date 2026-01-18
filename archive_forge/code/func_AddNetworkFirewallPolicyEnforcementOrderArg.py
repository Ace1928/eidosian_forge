from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddNetworkFirewallPolicyEnforcementOrderArg(parser):
    """Adds the --network-firewall-policy-enforcement-order flag."""
    parser.add_argument('--network-firewall-policy-enforcement-order', choices=_NETWORK_FIREWALL_POLICY_ENFORCEMENT_ORDER_CHOICES, metavar='NETWORK_FIREWALL_POLICY_ENFORCEMENT_ORDER', help='The Network Firewall Policy enforcement order of this network. If\n              not specified, defaults to AFTER_CLASSIC_FIREWALL.')