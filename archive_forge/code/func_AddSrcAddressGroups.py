from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddSrcAddressGroups(parser):
    """Adds a source address group to this rule."""
    parser.add_argument('--src-address-groups', type=arg_parsers.ArgList(), metavar='SOURCE_ADDRESS_GROUPS', required=False, help='Source address groups to match for this rule. Can only be specified if DIRECTION is ingress.')