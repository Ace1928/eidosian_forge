from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetStackType(ipv6_only_vpn_enabled=False):
    """Returns the flag for VPN gateway stack type.

  Args:
    ipv6_only_vpn_enabled: Whether to include IPV6_ONLY stack type.

  Return:
    An enum presents the stack type for the VPN gateway.
  """
    choices = {'IPV4_ONLY': 'Only IPv4 protocol is enabled on this VPN gateway.', 'IPV4_IPV6': 'Both IPv4 and IPv6 protocols are enabled on this VPN gateway.'}
    if ipv6_only_vpn_enabled:
        choices['IPV6_ONLY'] = 'Only IPv6 protocol is enabled on this VPN gateway.'
    return base.Argument('--stack-type', choices=choices, type=arg_utils.ChoiceToEnumName, help='      The stack type of the protocol(s) enabled on this VPN gateway.\n      If not provided, `IPV4_ONLY` will be used.\n      ')