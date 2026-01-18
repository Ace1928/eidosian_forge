from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddInternalIpv6RangeArg(parser):
    """Adds the --internal-ipv6-range flag."""
    parser.add_argument('--internal-ipv6-range', type=str, help='When enabling ULA internal IPv6, caller can optionally specify\n      the /48 range they want from the google defined ULA prefix fd20::/20.\n      ULA_IPV6_RANGE must be a valid /48 ULA IPv6 address and within the\n      fd20::/20. Operation will fail if the speficied /48 is already in used\n      by another resource. If the field is not speficied, then a /48 range\n      will be randomly allocated from fd20::/20 and returned via this field.')