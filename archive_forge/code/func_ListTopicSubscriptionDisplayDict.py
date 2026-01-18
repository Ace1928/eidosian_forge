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
def ListTopicSubscriptionDisplayDict(topic_subscription):
    """Returns a topic_subscription dict with additional fields."""
    result = resource_projector.MakeSerializable({'subscription': topic_subscription})
    subscription_ref = ParseSubscription(topic_subscription)
    result['projectId'] = subscription_ref.projectsId
    result['subscriptionId'] = subscription_ref.subscriptionsId
    return result