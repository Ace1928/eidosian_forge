from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _DeadLetterPolicy(self, dead_letter_topic, max_delivery_attempts):
    """Builds DeadLetterPolicy message from argument values.

    Args:
      dead_letter_topic (str): Topic for publishing dead messages.
      max_delivery_attempts (int): Threshold of failed deliveries before sending
        message to the dead letter topic.

    Returns:
      DeadLetterPolicy message or None.
    """
    if dead_letter_topic:
        return self.messages.DeadLetterPolicy(deadLetterTopic=dead_letter_topic, maxDeliveryAttempts=max_delivery_attempts)
    return None