from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddFilterCidrRangesArg(parser, is_for_update=False):
    """Adds args to specify filter CIDR ranges."""
    if is_for_update:
        cidr_ranges = parser.add_mutually_exclusive_group(help='Update the filter CIDR ranges of this packet mirroring.')
        cidr_ranges.add_argument('--add-filter-cidr-ranges', type=arg_parsers.ArgList(), metavar='CIDR_RANGE', help='List of filter CIDR ranges to add to the packet mirroring.')
        cidr_ranges.add_argument('--remove-filter-cidr-ranges', type=arg_parsers.ArgList(), metavar='CIDR_RANGE', help='List of filter CIDR ranges to remove from the packet mirroring.')
        cidr_ranges.add_argument('--set-filter-cidr-ranges', type=arg_parsers.ArgList(), metavar='CIDR_RANGE', help='        List of filter CIDR ranges to be mirrored on the packet mirroring.\n        ')
        cidr_ranges.add_argument('--clear-filter-cidr-ranges', action='store_true', default=None, help='        If specified, clear the existing filter CIDR ranges from the packet\n        mirroring.\n        ')
    else:
        parser.add_argument('--filter-cidr-ranges', type=arg_parsers.ArgList(), metavar='CIDR_RANGE', help='        List of IP CIDR ranges that apply as filters on the source or\n        destination IP in the IP header for packet mirroring traffic. All\n        traffic between the VM and the IPs listed here will be mirrored using\n        this configuration. This can be a Public IP as well.\n        If unspecified, the config applies to all traffic.\n        ')