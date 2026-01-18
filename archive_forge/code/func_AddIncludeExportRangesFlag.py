from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.network_connectivity import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddIncludeExportRangesFlag(parser, hide_include_export_ranges_flag):
    """Adds the --include-export-ranges argument to the given parser."""
    parser.add_argument('--include-export-ranges', required=False, type=arg_parsers.ArgList(), default=[], metavar='CIDR_RANGE', hidden=hide_include_export_ranges_flag, help='IP address range(s) to be allowed for subnets in VPC networks that are peered\n        through Network Connectivity Center peering.')