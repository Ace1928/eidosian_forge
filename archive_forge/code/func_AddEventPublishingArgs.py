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
def AddEventPublishingArgs(parser):
    """Adds an argument for an Eventarc channel and channel connection."""
    parser.add_argument('--event-id', required=True, help='An event id. The id of a published event.')
    parser.add_argument('--event-type', required=True, help='An event type. The event type of a published event.')
    parser.add_argument('--event-source', required=True, help='An event source. The event source of a published event.')
    parser.add_argument('--event-data', required=True, help='An event data. The event data of a published event.')
    parser.add_argument('--event-attributes', action=arg_parsers.UpdateAction, type=arg_parsers.ArgDict(), metavar='ATTRIBUTE=VALUE', help='Event attributes. The event attributes of a published event.This flag can be repeated to add more attributes.')