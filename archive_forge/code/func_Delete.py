from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.pubsub_apitools.pubsub_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
from the subscription.
def Delete(self, request, global_params=None):
    """Deletes the topic with the given name. Returns `NOT_FOUND` if the topic.
does not exist. After a topic is deleted, a new topic may be created with
the same name; this is an entirely new topic with none of the old
configuration or subscriptions. Existing subscriptions to this topic are
not deleted, but their `topic` field is set to `_deleted-topic_`.

      Args:
        request: (PubsubProjectsTopicsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('Delete')
    return self._RunMethod(config, request, global_params=global_params)