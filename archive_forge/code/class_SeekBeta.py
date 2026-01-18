from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import flags
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class SeekBeta(Seek):
    """Resets a subscription's backlog to a point in time or to a given snapshot."""

    def Run(self, args):
        flags.ValidateSubscriptionArgsUseUniverseSupportedFeatures(args)
        return _Run(args)