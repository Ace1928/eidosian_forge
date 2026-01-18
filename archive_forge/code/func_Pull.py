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
def Pull(self, request, global_params=None):
    """Pulls messages from the server. Returns an empty list if there are no.
messages available in the backlog. The server may return `UNAVAILABLE` if
there are too many concurrent pull requests pending for the given
subscription.

      Args:
        request: (PubsubProjectsSubscriptionsPullRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PullResponse) The response message.
      """
    config = self.GetMethodConfig('Pull')
    return self._RunMethod(config, request, global_params=global_params)