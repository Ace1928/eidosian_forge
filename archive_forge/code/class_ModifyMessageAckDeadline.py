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
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ModifyMessageAckDeadline(base.Command):
    """Modifies the ACK deadline for a specific Cloud Pub/Sub message."""
    detailed_help = {'DESCRIPTION': '          This method is useful to indicate that more time is needed to process\n          a message by the subscriber, or to make the message available for\n          redelivery if the processing was interrupted.'}

    @staticmethod
    def Args(parser):
        resource_args.AddSubscriptionResourceArg(parser, 'messages belong to.')
        flags.AddAckIdFlag(parser, 'modify the deadline for.')
        flags.AddAckDeadlineFlag(parser, required=True)

    def Run(self, args):
        result, ack_ids_and_failure_reasons = _Run(args, args.ack_ids, capture_failures=True)
        if ack_ids_and_failure_reasons:
            return ack_ids_and_failure_reasons
        return result