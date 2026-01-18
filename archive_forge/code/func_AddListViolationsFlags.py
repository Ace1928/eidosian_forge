from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddListViolationsFlags(parser):
    """Method to add list violations flags."""
    AddListWorkloadsFlags(parser)
    parser.add_argument('--workload', required=True, help='The parent workload of the Assured Workloads violations, provided as workload ID.')