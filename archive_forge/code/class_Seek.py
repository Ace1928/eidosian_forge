from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import flags
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Seek(base.Command):
    """Resets a subscription's backlog to a point in time or to a given snapshot."""

    @staticmethod
    def Args(parser):
        resource_args.AddSubscriptionResourceArg(parser, 'to affect.')
        flags.AddSeekFlags(parser)

    def Run(self, args):
        return _Run(args)