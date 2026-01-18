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
def Acknowledge(self, request, global_params=None):
    """Acknowledges the messages associated with the `ack_ids` in the.
`AcknowledgeRequest`. The Pub/Sub system can remove the relevant messages
from the subscription.

Acknowledging a message whose ack deadline has expired may succeed,
but such a message may be redelivered later. Acknowledging a message more
than once will not result in an error.

      Args:
        request: (PubsubProjectsSubscriptionsAcknowledgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('Acknowledge')
    return self._RunMethod(config, request, global_params=global_params)