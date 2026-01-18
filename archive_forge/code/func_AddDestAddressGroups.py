from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddDestAddressGroups(parser):
    """Adds a destination address group to this rule."""
    parser.add_argument('--dest-address-groups', type=arg_parsers.ArgList(), metavar='DEST_ADDRESS_GROUPS', required=False, help='Destination address groups to match for this rule. Can only be specified if DIRECTION is egress.')