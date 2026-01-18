from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetIpsecInternalAddressesFlag():
    """Adds ipsec-internal-addresses flag to the argparse.ArgumentParser."""
    return base.Argument('--ipsec-internal-addresses', required=False, type=arg_parsers.ArgList(max_length=1), metavar='ADDRESSES', help="      List of IP address range names that have been reserved for the interconnect\n      attachment (VLAN attachment). Use this option only for an interconnect\n      attachment that has its encryption option set as IPSEC. Currently only one\n      internal IP address range can be specified for each attachment.\n      When creating an HA VPN gateway for the interconnect attachment, if the\n      attachment is configured to use a regional internal IP address, then the VPN\n      gateway's IP address is allocated from the IP address range specified here.\n      If this field is not specified when creating the interconnect attachment,\n      then when creating any HA VPN gateways for this interconnect attachment,\n      the HA VPN gateway's IP address is allocated from a regional external IP\n      address pool.\n      ")