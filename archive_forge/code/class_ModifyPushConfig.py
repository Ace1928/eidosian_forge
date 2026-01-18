from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import flags
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ModifyPushConfig(base.Command):
    """Modifies the push configuration of a Cloud Pub/Sub subscription."""

    @classmethod
    def Args(cls, parser):
        _Args(parser)

    def Run(self, args):
        return _Run(args)