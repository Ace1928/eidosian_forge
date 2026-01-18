from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCustomerRouterIpv6InterfaceId(parser):
    """Adds customer router ipv6 interface id flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--customer-router-ipv6-interface-id', metavar='PEER_INTERFACE_ID', help='The `customer-router-ipv6-interface-id` field is not available.')