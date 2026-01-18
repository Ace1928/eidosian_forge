from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
def ProjectZonalMetadataAddMetadataFlags(parser):
    """Flags for adding/updating metadata on instance settings."""
    parser.add_argument('--metadata', default={}, type=arg_parsers.ArgDict(min_length=1), help='The project zonal metadata key-value pairs that you want to add or update\n\n', metavar='KEY=VALUE', required=True, action=arg_parsers.StoreOnceAction)
    parser.add_argument('--zone', help='The zone in which you want to add or update project zonal metadata\n\n', completer=compute_completers.ZonesCompleter, required=True)