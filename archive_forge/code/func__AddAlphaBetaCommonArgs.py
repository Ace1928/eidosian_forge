from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.core import properties
def _AddAlphaBetaCommonArgs(parser):
    """Add args and flags that are common to ALPHA and BETA tracks."""
    parser.add_argument('names', metavar='NAME', nargs='*', default=[], completer=completers.DiskTypesCompleter, help='If provided, show details for the specified names and/or URIs of resources.')
    parser.add_argument('--regexp', '-r', help='      A regular expression to filter the names of the results on. Any names\n      that do not match the entire regular expression will be filtered out.\n      ')
    parser.display_info.AddCacheUpdater(completers.DiskTypesCompleter)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--zones', metavar='ZONE', help='If provided, only zonal resources are shown. If arguments are provided, only resources from the given zones are shown.', type=arg_parsers.ArgList())
    scope.add_argument('--regions', metavar='REGION', help='If provided, only regional resources are shown. If arguments are provided, only resources from the given regions are shown.', type=arg_parsers.ArgList())
    parser.display_info.AddFormat('\n        table(\n          name,\n          location():label=LOCATION,\n          location_scope():label=SCOPE,\n          validDiskSize:label=VALID_DISK_SIZES\n        )\n  ')