from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddFilterProtocolsArg(parser, is_for_update=False):
    """Adds args to specify filter IP protocols."""
    if is_for_update:
        protocols = parser.add_mutually_exclusive_group(help='Update the filter protocols of this packet mirroring.')
        protocols.add_argument('--add-filter-protocols', type=arg_parsers.ArgList(element_type=str), metavar='PROTOCOL', help='        List of filter IP protocols to add to the packet mirroring.\n        PROTOCOL can be one of tcp, udp, icmp, esp, ah, ipip, sctp, or an IANA\n        protocol number.\n        ')
        protocols.add_argument('--remove-filter-protocols', type=arg_parsers.ArgList(element_type=str), metavar='PROTOCOL', help='        List of filter IP protocols to remove from the packet mirroring.\n        PROTOCOL can be one of tcp, udp, icmp, esp, ah, ipip, sctp, or an IANA\n        protocol number.\n        ')
        protocols.add_argument('--set-filter-protocols', type=arg_parsers.ArgList(element_type=str), metavar='PROTOCOL', help='        List of filter IP protocols to be mirrored on the packet mirroring.\n        PROTOCOL can be one of tcp, udp, icmp, esp, ah, ipip, sctp, or an IANA\n        protocol number.\n        ')
        protocols.add_argument('--clear-filter-protocols', action='store_true', default=None, help='        If specified, clear the existing filter IP protocols from the packet\n        mirroring.\n        ')
    else:
        parser.add_argument('--filter-protocols', type=arg_parsers.ArgList(element_type=str), metavar='PROTOCOL', help='        List of IP protocols that apply as filters for packet mirroring traffic.\n        If unspecified, the packet mirroring applies to all traffic.\n        PROTOCOL can be one of tcp, udp, icmp, esp, ah, ipip, sctp, or an IANA\n        protocol number.\n        ')