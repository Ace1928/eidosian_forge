from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2NotificationConfig(_messages.Message):
    """Defines notification behavior.

  Enums:
    MessageFormatValueValuesEnum: Format of message.

  Fields:
    messageFormat: Format of message.
    topic: Name of the Pub/Sub topic to publish conversation events like
      CONVERSATION_STARTED as serialized ConversationEvent protos. For
      telephony integration to receive notification, make sure either this
      topic is in the same project as the conversation or you grant
      `service-@gcp-sa-dialogflow.iam.gserviceaccount.com` the `Dialogflow
      Service Agent` role in the topic project. For chat integration to
      receive notification, make sure API caller has been granted the
      `Dialogflow Service Agent` role for the topic. Format:
      `projects//locations//topics/`.
  """

    class MessageFormatValueValuesEnum(_messages.Enum):
        """Format of message.

    Values:
      MESSAGE_FORMAT_UNSPECIFIED: If it is unspecified, PROTO will be used.
      PROTO: Pub/Sub message will be serialized proto.
      JSON: Pub/Sub message will be json.
    """
        MESSAGE_FORMAT_UNSPECIFIED = 0
        PROTO = 1
        JSON = 2
    messageFormat = _messages.EnumField('MessageFormatValueValuesEnum', 1)
    topic = _messages.StringField(2)