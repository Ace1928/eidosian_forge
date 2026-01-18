from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddAuthorizedNetworksFlag(parser):
    """Adds a `--authorized-networks` flag."""
    cidr_validator = arg_parsers.RegexpValidator(_CIDR_REGEX, "Must be specified in CIDR notation, also known as 'slash' notation (e.g. 192.168.100.0/24).")
    help_text = "    List of external networks that are allowed to connect to the instance.\n    Specify values in CIDR notation, also known as 'slash' notation\n    (e.g.192.168.100.0/24).\n    "
    parser.add_argument('--authorized-networks', type=arg_parsers.ArgList(min_length=1, element_type=cidr_validator), metavar='NETWORK', default=[], help=help_text)