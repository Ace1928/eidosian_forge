from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddCreatePdpArgsToParser(parser, support_ipv6_pdp):
    """Adds flags for public delegated prefixes create command."""
    parent_prefix_args = parser.add_mutually_exclusive_group(required=True)
    parent_prefix_args.add_argument('--public-advertised-prefix', help='Public advertised prefix that this delegated prefix is created from.')
    parent_prefix_args.add_argument('--public-delegated-prefix', help='Regional Public delegated prefix that this delegated prefix is created from.')
    parser.add_argument('--range', required=True, help='IP range from this public delegated prefix that should be delegated, in CIDR format. It must be smaller than parent public advertised prefix range.')
    parser.add_argument('--description', help='Description of this public delegated prefix.')
    parser.add_argument('--enable-live-migration', action='store_true', default=None, help='Specify if this public delegated prefix is meant to be live migrated.')
    if support_ipv6_pdp:
        parser.add_argument('--mode', choices=['EXTERNAL_IPV6_FORWARDING_RULE_CREATION', 'DELEGATION'], help='Specifies the mode of this IPv6 PDP.')
        parser.add_argument('--allocatable-prefix-length', help='The allocatable prefix length supported by this PDP.')