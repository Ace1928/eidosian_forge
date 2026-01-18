from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
def AddArgsForEgress(parser, ruleset_parser, for_update=False):
    """Adds arguments for egress firewall create or update subcommands."""
    min_length = 0 if for_update else 1
    if not for_update:
        ruleset_parser.add_argument('--action', choices=['ALLOW', 'DENY'], type=lambda x: x.upper(), help='        The action for the firewall rule: whether to allow or deny matching\n        traffic. If specified, the flag `--rules` must also be specified.\n        ')
    rules_help = '      A list of protocols and ports to which the firewall rule will apply.\n\n      PROTOCOL is the IP protocol whose traffic will be checked.\n      PROTOCOL can be either the name of a well-known protocol\n      (e.g., tcp or icmp) or the IP protocol number.\n      A list of IP protocols can be found at\n      http://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml\n\n      A port or port range can be specified after PROTOCOL to which the\n      firewall rule apply on traffic through specific ports. If no port\n      or port range is specified, connections through all ranges are applied.\n      TCP and UDP rules must include a port or port range.\n      '
    if for_update:
        rules_help += '\n      Setting this will override the current values.\n      '
    else:
        rules_help += '\n      If specified, the flag --action must also be specified.\n\n      For example, the following will create a rule that blocks TCP\n      traffic through port 80 and ICMP traffic:\n\n        $ {command} MY-RULE --action deny --rules tcp:80,icmp\n      '
    parser.add_argument('--rules', metavar=ALLOWED_METAVAR, type=arg_parsers.ArgList(min_length=min_length), help=rules_help, required=False)
    if not for_update:
        parser.add_argument('--direction', choices=['INGRESS', 'EGRESS', 'IN', 'OUT'], type=lambda x: x.upper(), help="        If direction is NOT specified, then default is to apply on incoming\n        traffic. For outbound traffic, it is NOT supported to specify\n        source-tags.\n\n        For convenience, 'IN' can be used to represent ingress direction and\n        'OUT' can be used to represent egress direction.\n        ")
    parser.add_argument('--priority', type=int, help='      This is an integer between 0 and 65535, both inclusive. When NOT\n      specified, the value assumed is 1000. Relative priority determines\n      precedence of conflicting rules: lower priority values imply higher\n      precedence. DENY rules take precedence over ALLOW rules having equal\n      priority.\n      ')
    destination_ranges_help = '      The firewall rule will apply to traffic that has destination IP address\n      in these IP address block list. The IP address blocks must be specified\n      in CIDR format:\n      link:http://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing[].\n      '
    if for_update:
        destination_ranges_help += '\n      Setting this will override the existing destination ranges for the\n      firewall. The following will clear the existing destination ranges:\n\n        $ {command} MY-RULE --destination-ranges\n      '
    else:
        destination_ranges_help += '\n      If --destination-ranges is NOT provided, then this\n      flag will default to 0.0.0.0/0, allowing all IPv4 destinations. Multiple\n      IP address blocks can be specified if they are separated by commas.\n      '
    parser.add_argument('--destination-ranges', default=None if for_update else [], metavar='CIDR_RANGE', type=arg_parsers.ArgList(min_length=min_length), help=destination_ranges_help)