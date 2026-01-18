from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import googlecloudsdk
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddDestinationRunRegionArg(parser, required=False):
    """Adds an argument for the trigger's destination Cloud Run service's region."""
    parser.add_argument('--destination-run-region', required=required, help='Region in which the destination Cloud Run service can be found. If not specified, it is assumed that the service is in the same region as the trigger.')