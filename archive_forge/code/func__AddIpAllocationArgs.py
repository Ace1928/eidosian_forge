from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddIpAllocationArgs(parser):
    """Adds a mutually exclusive group to specify IP allocation options."""
    ip_allocation = parser.add_mutually_exclusive_group(required=False)
    ip_allocation.add_argument('--auto-allocate-nat-external-ips', help='Automatically allocate external IP addresses for Cloud NAT', action='store_true', default=False)
    IP_ADDRESSES_ARG.AddArgument(parser, mutex_group=ip_allocation, cust_metavar='IP_ADDRESS')