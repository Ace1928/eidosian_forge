from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddIpAndRangeArgsForUpdate(parser):
    """Adds argument to specify source NAT IP Addresses when updating a rule."""
    ACTIVE_RANGES_ARG.AddArgument(parser, cust_metavar='SUBNETWORK')
    ACTIVE_IPS_ARG_OPTIONAL.AddArgument(parser, cust_metavar='IP_ADDRESS')
    drain_ip_mutex = parser.add_mutually_exclusive_group(required=False)
    drain_ip_mutex.add_argument('--clear-source-nat-drain-ips', help='Clear drained IPs from the rule', action='store_true', default=None)
    DRAIN_IPS_ARG.AddArgument(parser, mutex_group=drain_ip_mutex, cust_metavar='IP_ADDRESS')
    drain_range_mutex = parser.add_mutually_exclusive_group(required=False)
    drain_range_mutex.add_argument('--clear-source-nat-drain-ranges', help='Clear drained ranges from the rule', action='store_true', default=None)
    DRAIN_RANGES_ARG.AddArgument(parser, mutex_group=drain_range_mutex, cust_metavar='SUBNETWORK')