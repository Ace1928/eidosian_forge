from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def ListSubscriptions(self, topic_ref, page_size=100):
    """Lists Subscriptions for a given topic.

    Args:
      topic_ref (Resource): Resource reference to Topic to list subscriptions
        from.
      page_size (int): the number of entries in each batch (affects requests
        made, but not the yielded results).

    Returns:
      A generator of Subscriptions for the Topic..
    """
    list_req = self.messages.PubsubProjectsTopicsSubscriptionsListRequest(topic=topic_ref.RelativeName(), pageSize=page_size)
    list_subs_service = self.client.projects_topics_subscriptions
    return list_pager.YieldFromList(list_subs_service, list_req, batch_size=page_size, field='subscriptions', batch_size_attribute='pageSize')