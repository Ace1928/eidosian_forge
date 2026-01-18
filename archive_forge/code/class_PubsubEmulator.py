from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
class PubsubEmulator(util.Emulator):
    """Represents the ability to start and route pubsub emulator."""

    def Start(self, port):
        args = util.AttrDict({'host_port': {'host': '::1', 'port': port}})
        return Start(args, self._GetLogNo())

    @property
    def prefixes(self):
        return ['google.pubsub.v1.Publisher', 'google.pubsub.v1.Subscriber', 'google.pubsub.v1.AcknowledgeRequest', 'google.pubsub.v1.DeleteSubscriptionRequest', 'google.pubsub.v1.DeleteTopicRequest', 'google.pubsub.v1.GetSubscriptionRequest', 'google.pubsub.v1.GetTopicRequest', 'google.pubsub.v1.ListSubscriptionsRequest', 'google.pubsub.v1.ListSubscriptionsResponse', 'google.pubsub.v1.ListTopicSubscriptionsRequest', 'google.pubsub.v1.ListTopicSubscriptionsResponse', 'google.pubsub.v1.ListTopicsRequest', 'google.pubsub.v1.ListTopicsResponse', 'google.pubsub.v1.ModifyAckDeadlineRequest', 'google.pubsub.v1.ModifyPushConfigRequest', 'google.pubsub.v1.PublishRequest', 'google.pubsub.v1.PublishResponse', 'google.pubsub.v1.PubsubMessage', 'google.pubsub.v1.PullRequest', 'google.pubsub.v1.PullResponse', 'google.pubsub.v1.PushConfig', 'google.pubsub.v1.ReceivedMessage', 'google.pubsub.v1.Subscription', 'google.pubsub.v1.Topic']

    @property
    def service_name(self):
        return PUBSUB

    @property
    def emulator_title(self):
        return PUBSUB_TITLE

    @property
    def emulator_component(self):
        return 'pubsub-emulator'