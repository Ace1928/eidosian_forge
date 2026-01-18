from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.api_lib.util import exceptions as exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import times
import six
def SubscriptionDisplayDict(subscription):
    """Creates a serializable from a Cloud Pub/Sub Subscription op for display.

  Args:
    subscription: (Cloud Pub/Sub Subscription) Subscription to be serialized.
  Returns:
    A serialized object representing a Cloud Pub/Sub Subscription
    operation (create, delete, update).
  """
    push_endpoint = ''
    subscription_type = 'pull'
    if subscription.pushConfig:
        if subscription.pushConfig.pushEndpoint:
            push_endpoint = subscription.pushConfig.pushEndpoint
            subscription_type = 'push'
    return {'subscriptionId': subscription.name, 'topic': subscription.topic, 'type': subscription_type, 'pushEndpoint': push_endpoint, 'ackDeadlineSeconds': subscription.ackDeadlineSeconds, 'retainAckedMessages': bool(subscription.retainAckedMessages), 'messageRetentionDuration': subscription.messageRetentionDuration, 'enableExactlyOnceDelivery': subscription.enableExactlyOnceDelivery}