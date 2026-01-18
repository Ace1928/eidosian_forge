from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddEncryption(parser):
    """Adds encryption flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--encryption', required=False, choices=_ENCRYPTION_CHOICES, help='      Indicates the user-supplied encryption option for this interconnect\n      attachment (VLAN attachment).\n\n      Possible values are:\n\n      `NONE` - This is the default value, which means the interconnect attachment\n      carries unencrypted traffic. VMs can send traffic to or\n      receive traffic from such interconnect attachment.\n\n      `IPSEC` - The interconnect attachment carries only traffic that is encrypted\n      by an IPsec device; for example, an HA VPN gateway or third-party\n      IPsec VPN. VMs cannot directly send traffic to or receive traffic from such\n      an interconnect attachment. To use HA VPN over Cloud Interconnect,\n      the interconnect attachment must be created with this option.\n\n      ')