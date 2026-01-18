from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import lite_util
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class Subscribe(base.Command):
    """Stream messages from a Pub/Sub Lite subscription."""
    detailed_help = {'DESCRIPTION': '          Streams messages from a Pub/Sub Lite subscription. This command\n          requires Python 3.6 or greater, and requires the grpcio Python package\n          to be installed.\n\n          For MacOS, Linux, and Cloud Shell users, to install the gRPC client\n          libraries, run:\n\n            $ sudo pip3 install grpcio\n            $ export CLOUDSDK_PYTHON_SITEPACKAGES=1\n      ', 'EXAMPLES': '          To subscribe to a Pub/Sub Lite subscription and automatically\n          acknowledge messages, run:\n\n            $ {command} mysubscription --location=us-central1-a --auto-ack\n\n          To subscribe to specific partitions in a subscription, run:\n\n            $ {command} mysubscription --location=us-central1-a --partitions=0,1,2\n      '}

    @staticmethod
    def Args(parser):
        resource_args.AddResourceArgToParser(parser, resource_path='pubsub.lite_subscription', required=True, help_text='The Pub/Sub Lite subscription to receive messages from.')
        parser.add_argument('--num-messages', type=arg_parsers.BoundedInt(1, 1000), default=1, help='The number of messages to stream before exiting. This value must\n        be less than or equal to 1000.')
        parser.add_argument('--auto-ack', action='store_true', default=False, help='Automatically ACK every message received on this subscription.')
        parser.add_argument('--partitions', metavar='INT', type=arg_parsers.ArgList(element_type=int), help='The partitions this subscriber should connect to to receive\n        messages. If empty, partitions will be automatically assigned.')

    def Run(self, args):
        lite_util.RequirePython36('gcloud pubsub lite-subscriptions subscribe')
        try:
            from googlecloudsdk.api_lib.pubsub import lite_subscriptions
        except ImportError:
            raise lite_util.NoGrpcInstalled()
        log.out.Print('Initializing the Subscriber stream... This may take up to 30 seconds.')
        printer = resource_printer.Printer(args.format or MESSAGE_FORMAT)
        with lite_subscriptions.SubscriberClient(args.CONCEPTS.subscription.Parse(), args.partitions or [], args.num_messages, args.auto_ack) as subscriber_client:
            received = 0
            while received < args.num_messages:
                message = subscriber_client.Pull()
                if message:
                    splits = message.message_id.split(',')
                    message.message_id = 'Partition: {}, Offset: {}'.format(splits[0], splits[1])
                    printer.Print([message])
                    received += 1