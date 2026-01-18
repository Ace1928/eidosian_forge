from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
def AddDescribeProjectZonalMetadataFlags(parser):
    parser.add_argument('--zone', help='Zone for project zonal metadata', completer=compute_completers.ZonesCompleter, required=True)