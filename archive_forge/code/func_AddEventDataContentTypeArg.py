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
def AddEventDataContentTypeArg(parser, release_track, required=False, hidden=False):
    """Adds an argument for the trigger's event data content type."""
    if release_track == base.ReleaseTrack.GA:
        parser.add_argument('--event-data-content-type', hidden=hidden, required=required, help="Depending on the event provider, you can specify the encoding of the event data payload that will be delivered to your destination, to either be encoded in ``application/json'' or ``application/protobuf''. The default encoding is ``application/json''. Note that for custom sources or third-party providers, or for direct events from Cloud Pub/Sub, this formatting option is not supported.")