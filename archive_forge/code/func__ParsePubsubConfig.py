from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParsePubsubConfig(topic_name, message_format=None, service_account=None):
    """Parse and create PubsubConfig message."""
    message_format_enums = _MESSAGES.PubsubConfig.MessageFormatValueValuesEnum
    if message_format == 'protobuf':
        parsed_message_format = message_format_enums.PROTOBUF
    else:
        parsed_message_format = message_format_enums.JSON
    return _MESSAGES.PubsubConfig(messageFormat=parsed_message_format, serviceAccountEmail=service_account, topic=topic_name)