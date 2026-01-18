from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddProducerAcceptList(parser):
    """Add support for --producer-accept-list flag."""
    parser.add_argument('--producer-accept-list', type=arg_parsers.ArgList(), metavar='ACCEPT_LIST', default=None, help='      Projects that are allowed to connect to this network attachment.\n      ')