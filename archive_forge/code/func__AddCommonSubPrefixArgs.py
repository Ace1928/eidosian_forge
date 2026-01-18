from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddCommonSubPrefixArgs(parser, verb):
    """Adds common flags for delegate sub prefixes create/delete commands."""
    parser.add_argument('name', help='Name of the delegated sub prefix to {}.'.format(verb))
    PUBLIC_DELEGATED_PREFIX_FLAG_ARG.AddArgument(parser, operation_type='{} the delegate sub prefix for'.format(verb))