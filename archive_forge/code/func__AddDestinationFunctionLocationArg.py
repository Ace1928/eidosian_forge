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
def _AddDestinationFunctionLocationArg(parser, required=False):
    """Adds an argument for the trigger's destination Function location."""
    parser.add_argument('--destination-function-location', required=required, help='Location that the destination Function is running in. If not specified, it is assumed that the Function is in the same location as the trigger.')