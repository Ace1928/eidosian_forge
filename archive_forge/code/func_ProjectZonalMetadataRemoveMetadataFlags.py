from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
def ProjectZonalMetadataRemoveMetadataFlags(parser):
    """Flags for removing metadata on instance settings."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', default=False, help='If provided, all project zonal metadata entries are removed from VM instances in the zone.', action='store_true')
    group.add_argument('--keys', default={}, type=arg_parsers.ArgList(min_length=1), metavar='KEY', help='The keys for which you want to remove project zonal metadata\n\n')
    parser.add_argument('--zone', help='The zone in which you want to remove project zonal metadata\n\n', completer=compute_completers.ZonesCompleter, required=True)