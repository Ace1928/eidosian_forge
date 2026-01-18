from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudiot import devices
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.command_lib.iot import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import times
import six
def ParseEventNotificationConfig(event_notification_configs, messages=None):
    """Creates a list of EventNotificationConfigs from args."""
    messages = messages or registries.GetMessagesModule()
    if event_notification_configs:
        configs = []
        for config in event_notification_configs:
            topic_ref = ParsePubsubTopic(config['topic'])
            configs.append(messages.EventNotificationConfig(pubsubTopicName=topic_ref.RelativeName(), subfolderMatches=config.get('subfolder', None)))
        return configs
    return None