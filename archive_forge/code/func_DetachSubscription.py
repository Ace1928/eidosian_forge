from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def DetachSubscription(self, subscription_ref):
    """Detaches the subscription from its topic.

    Args:
      subscription_ref (Resource): Resource reference to the Subscription to
        detach.

    Returns:
      Empty: An empty response message.
    """
    detach_req = self.messages.PubsubProjectsSubscriptionsDetachRequest(subscription=subscription_ref.RelativeName())
    return self._subscriptions_service.Detach(detach_req)