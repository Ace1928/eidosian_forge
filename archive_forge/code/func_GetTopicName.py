from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetTopicName(args):
    """Get the topic name based on project and topic_project flags."""
    if args.add_topic:
        topic_ref = args.CONCEPTS.add_topic.Parse()
    elif args.remove_topic:
        topic_ref = args.CONCEPTS.remove_topic.Parse()
    else:
        topic_ref = args.CONCEPTS.update_topic.Parse()
    return topic_ref.RelativeName()