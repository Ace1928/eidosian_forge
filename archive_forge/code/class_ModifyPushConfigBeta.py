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
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ModifyPushConfigBeta(ModifyPushConfig):
    """Modifies the push configuration of a Cloud Pub/Sub subscription."""

    @classmethod
    def Args(cls, parser):
        _Args(parser)

    def Run(self, args):
        legacy_output = properties.VALUES.pubsub.legacy_output.GetBool()
        return _Run(args, legacy_output=legacy_output)