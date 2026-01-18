from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _UpdateConfigWithArgs(pubsub_config, args):
    message_format = args.message_format or _GetMessageFormatString(pubsub_config)
    service_account = args.service_account or getattr(pubsub_config, 'serviceAccountEmail')
    topic_name = pubsub_config.topic
    return _ParsePubsubConfig(topic_name, message_format, service_account)