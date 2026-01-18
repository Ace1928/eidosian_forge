from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddDrainNatIpsArgument(parser):
    drain_ips_group = parser.add_mutually_exclusive_group(required=False)
    DRAIN_NAT_IP_ADDRESSES_ARG.AddArgument(parser, mutex_group=drain_ips_group)
    drain_ips_group.add_argument('--clear-nat-external-drain-ip-pool', action='store_true', default=False, help='Clear the drained NAT IPs')