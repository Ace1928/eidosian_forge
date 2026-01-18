from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddMatcherAndNetworkMatcher(parser, required=True):
    """Adds the matcher arguments to the argparse."""
    matcher = parser.add_group(required=required, help='Security policy rule matcher.')
    matcher.add_argument('--src-ip-ranges', type=arg_parsers.ArgList(), metavar='SRC_IP_RANGE', help='The source IPs/IP ranges to match for this rule. To match all IPs specify *.')
    matcher.add_argument('--expression', help='The Cloud Armor rules language expression to match for this rule.')
    matcher.add_argument('--network-user-defined-fields', type=arg_parsers.ArgList(), metavar='NAME;VALUE:VALUE:...', help='Each element names a defined field and lists the matching values for that field.')
    matcher.add_argument('--network-src-ip-ranges', type=arg_parsers.ArgList(), metavar='SRC_IP_RANGE', help='The source IPs/IP ranges to match for this rule. To match all IPs specify *.')
    matcher.add_argument('--network-dest-ip-ranges', type=arg_parsers.ArgList(), metavar='DEST_IP_RANGE', help='The destination IPs/IP ranges to match for this rule. To match all IPs specify *.')
    matcher.add_argument('--network-ip-protocols', type=arg_parsers.ArgList(), metavar='IP_PROTOCOL', help='The IP protocols to match for this rule. Each element can be an 8-bit unsigned decimal number (e.g. "6"), range (e.g."253-254"), or one of the following protocol names: "tcp", "udp", "icmp", "esp", "ah", "ipip", or "sctp". To match all protocols specify *.')
    matcher.add_argument('--network-src-ports', type=arg_parsers.ArgList(), metavar='SRC_PORT', help='The source ports to match for this rule. Each element can be an 16-bit unsigned decimal number (e.g. "80") or range (e.g."0-1023"), To match all source ports specify *.')
    matcher.add_argument('--network-dest-ports', type=arg_parsers.ArgList(), metavar='DEST_PORT', help='The destination ports to match for this rule. Each element can be an 16-bit unsigned decimal number (e.g. "80") or range (e.g."0-1023"), To match all destination ports specify *.')
    matcher.add_argument('--network-src-region-codes', type=arg_parsers.ArgList(), metavar='SRC_REGION_CODE', help='The two letter ISO 3166-1 alpha-2 country code associated with the source IP address to match for this rule. To match all region codes specify *.')
    matcher.add_argument('--network-src-asns', type=arg_parsers.ArgList(), metavar='SRC_ASN', help='BGP Autonomous System Number associated with the source IP address to match for this rule.')