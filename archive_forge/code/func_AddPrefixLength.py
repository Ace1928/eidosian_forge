from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddPrefixLength(parser):
    """Adds the prefix-length flag."""
    parser.add_argument('--prefix-length', type=arg_parsers.BoundedInt(lower_bound=8, upper_bound=96), help='      The prefix length of the IP range. If the address is an IPv4 address, it\n      must be a value between 8 and 30 inclusive. If the address is an IPv6\n      address, the only allowed value is 96. If not present, it means the\n      address field is a single IP address.\n\n      This field is not applicable to external IPv4 addresses or global IPv6\n      addresses.\n      ')