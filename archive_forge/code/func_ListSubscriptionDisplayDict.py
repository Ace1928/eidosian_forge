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
def ListSubscriptionDisplayDict(subscription):
    """Returns a subscription dict with additional fields."""
    result = resource_projector.MakeSerializable(subscription)
    result['type'] = 'PUSH' if subscription.pushConfig.pushEndpoint else 'PULL'
    subscription_ref = ParseSubscription(subscription.name)
    result['projectId'] = subscription_ref.projectsId
    result['subscriptionId'] = subscription_ref.subscriptionsId
    topic_info = ParseTopic(subscription.topic)
    result['topicId'] = topic_info.topicsId
    return result