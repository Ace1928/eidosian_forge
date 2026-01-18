from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddCreatePapArgsToParser(parser):
    """Adds public advertised prefixes create related flags to parser."""
    parser.add_argument('--range', required=True, help='IP range allocated to this public advertised prefix, in CIDR format.')
    parser.add_argument('--dns-verification-ip', required=True, help='IP address to use for verification. It must be within the IP range specified in --range.')
    parser.add_argument('--description', help='Description of this public advertised prefix.')
    choices = ['GLOBAL', 'REGIONAL']
    parser.add_argument('--pdp-scope', choices=choices, help='Specifies how child public delegated prefix will be scoped.')