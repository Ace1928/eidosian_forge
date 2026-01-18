from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ListSubscriptionsBeta(ListSubscriptions):
    """Lists Cloud Pub/Sub subscriptions from a given topic."""

    def Run(self, args):
        legacy_output = properties.VALUES.pubsub.legacy_output.GetBool()
        return _Run(args, legacy_output=legacy_output)