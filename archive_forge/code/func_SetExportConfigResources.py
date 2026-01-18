from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def SetExportConfigResources(args, psl, project, location, export_config):
    """Sets fully qualified resource paths for an ExportConfig."""
    if args.export_pubsub_topic:
        topic = args.export_pubsub_topic
        if not topic.startswith(PROJECTS_RESOURCE_PATH):
            topic = '{}{}/{}{}'.format(PROJECTS_RESOURCE_PATH, project, TOPICS_RESOURCE_PATH, topic)
        export_config.pubsubConfig = psl.PubSubConfig(topic=topic)
    if args.export_dead_letter_topic:
        topic = args.export_dead_letter_topic
        if not topic.startswith(PROJECTS_RESOURCE_PATH):
            topic = '{}{}/{}{}/{}{}'.format(PROJECTS_RESOURCE_PATH, project, LOCATIONS_RESOURCE_PATH, location, TOPICS_RESOURCE_PATH, topic)
        export_config.deadLetterTopic = topic