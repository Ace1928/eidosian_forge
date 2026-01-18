from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_ex
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import flags
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ModifyMessageAckDeadlineBeta(ModifyMessageAckDeadline):
    """Modifies the ACK deadline for a specific Cloud Pub/Sub message."""

    @staticmethod
    def Args(parser):
        resource_args.AddSubscriptionResourceArg(parser, 'messages belong to.')
        flags.AddAckIdFlag(parser, 'modify the deadline for.', add_deprecated=True)
        flags.AddAckDeadlineFlag(parser, required=True)

    def Run(self, args):
        ack_ids = flags.ParseAckIdsArgs(args)
        legacy_output = properties.VALUES.pubsub.legacy_output.GetBool()
        result, ack_ids_and_failure_reasons = _Run(args, ack_ids, legacy_output=legacy_output, capture_failures=True)
        if ack_ids_and_failure_reasons:
            return ack_ids_and_failure_reasons
        return result