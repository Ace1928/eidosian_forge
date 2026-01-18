from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetGatewayIpVersion():
    """Returns the flag for VPN gateway IP version.

  Return:
    An enum presents the gateway IP version for the VPN gateway.
  """
    return base.Argument('--gateway-ip-version', choices={'IPV4': 'Every HA-VPN gateway interface is configured with an IPv4 address.', 'IPV6': 'Every HA-VPN gateway interface is configured with an IPv6 address.'}, type=arg_utils.ChoiceToEnumName, help='      IP version of the HA VPN gateway. You must specify either IPv4 or IPv6. If\n      you do not specify this field, every HA VPN gateway interface will be\n      configured with an IPv4 address.\n      ')