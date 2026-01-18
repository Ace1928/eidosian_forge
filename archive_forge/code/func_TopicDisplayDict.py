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
def TopicDisplayDict(topic):
    """Creates a serializable from a Cloud Pub/Sub Topic operation for display.

  Args:
    topic: (Cloud Pub/Sub Topic) Topic to be serialized.
  Returns:
    A serialized object representing a Cloud Pub/Sub Topic
    operation (create, delete).
  """
    topic_display_dict = resource_projector.MakeSerializable(topic)
    topic_display_dict['topicId'] = topic.name
    del topic_display_dict['name']
    return topic_display_dict