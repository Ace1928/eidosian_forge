from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddEnableDataCacheFlag(parser):
    """Adds a --enable-data-cache flag to the given parser."""
    parser.add_argument('--enable-data-cache', hidden=True, required=False, action='store_true', dest='enable_data_cache', help='Enable use of data cache for accelerated read performance. This flag is only available for Enterprise Plus edition instances.')