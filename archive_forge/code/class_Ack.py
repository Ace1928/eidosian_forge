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
class Ack(base.Command):
    """Acknowledges one or more messages on the specified subscription."""
    detailed_help = {'DESCRIPTION': "          Acknowledges one or more messages as having been successfully received.\n          If a delivered message is not acknowledged within the Subscription's\n          ack deadline, Cloud Pub/Sub will attempt to deliver it again.\n\n          To automatically acknowledge messages when pulling from a Subscription,\n          you can use the `--auto-ack` flag on `gcloud pubsub subscriptions pull`.\n      "}

    @staticmethod
    def Args(parser):
        resource_args.AddSubscriptionResourceArg(parser, 'to ACK messages on.')
        flags.AddAckIdFlag(parser, 'acknowledge.')

    def Run(self, args):
        result, ack_ids_and_failure_reasons = _Run(args, args.ack_ids, capture_failures=True)
        if ack_ids_and_failure_reasons:
            return ack_ids_and_failure_reasons
        return result