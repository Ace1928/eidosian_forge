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
def AddTransportTopicResourceArg(parser, required=False):
    """Adds a resource argument for a customer-provided transport topic."""
    resource_spec = concepts.ResourceSpec('pubsub.projects.topics', resource_name='Pub/Sub topic', topicsId=TransportTopicAttributeConfig(), projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG)
    concept_parser = concept_parsers.ConceptParser.ForResource('--transport-topic', resource_spec, "The Cloud Pub/Sub topic to use for the trigger's transport intermediary. This feature is currently only available for triggers of event type ``google.cloud.pubsub.topic.v1.messagePublished''. The topic must be in the same project as the trigger. If not specified, a transport topic will be created.", required=required)
    concept_parser.AddToParser(parser)