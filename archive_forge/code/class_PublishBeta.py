from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import flags
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import http_encoding
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class PublishBeta(Publish):
    """Publishes a message to the specified topic."""

    @classmethod
    def Args(cls, parser):
        resource_args.AddTopicResourceArg(parser, 'to publish messages to.')
        flags.AddPublishMessageFlags(parser, add_deprecated=True)

    def Run(self, args):
        message_body = flags.ParseMessageBody(args)
        legacy_output = properties.VALUES.pubsub.legacy_output.GetBool()
        return _Run(args, message_body, legacy_output=legacy_output)