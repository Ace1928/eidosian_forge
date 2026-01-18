from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _PubsubExportConfig(self, topic, region):
    """Builds PubsubExportConfig message from argument values.

    Args:
      topic (str): The Pubsub topic to which to publish messages.
      region (str): The Cloud region to which to publish messages.

    Returns:
      PubsubExportConfig message or None
    """
    if topic:
        return self.messages.PubSubExportConfig(topic=topic, region=region)
    return None