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
def AddClearDestinationRunPathArg(parser):
    """Adds an argument for clearing the trigger's Cloud Run destination path."""
    parser.add_argument('--clear-destination-run-path', action='store_true', help='Clear the relative path on the destination Cloud Run service to which the events for the trigger should be sent.')