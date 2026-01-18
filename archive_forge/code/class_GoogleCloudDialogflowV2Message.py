from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2Message(_messages.Message):
    """Represents a message posted into a conversation.

  Enums:
    ParticipantRoleValueValuesEnum: Output only. The role of the participant.

  Fields:
    content: Required. The message content.
    createTime: Output only. The time when the message was created in Contact
      Center AI.
    languageCode: Optional. The message language. This should be a
      [BCP-47](https://www.rfc-editor.org/rfc/bcp/bcp47.txt) language tag.
      Example: "en-US".
    messageAnnotation: Output only. The annotation for the message.
    name: Optional. The unique identifier of the message. Format:
      `projects//locations//conversations//messages/`.
    participant: Output only. The participant that sends this message.
    participantRole: Output only. The role of the participant.
    sendTime: Optional. The time when the message was sent.
    sentimentAnalysis: Output only. The sentiment analysis result for the
      message.
  """

    class ParticipantRoleValueValuesEnum(_messages.Enum):
        """Output only. The role of the participant.

    Values:
      ROLE_UNSPECIFIED: Participant role not set.
      HUMAN_AGENT: Participant is a human agent.
      AUTOMATED_AGENT: Participant is an automated agent, such as a Dialogflow
        agent.
      END_USER: Participant is an end user that has called or chatted with
        Dialogflow services.
    """
        ROLE_UNSPECIFIED = 0
        HUMAN_AGENT = 1
        AUTOMATED_AGENT = 2
        END_USER = 3
    content = _messages.StringField(1)
    createTime = _messages.StringField(2)
    languageCode = _messages.StringField(3)
    messageAnnotation = _messages.MessageField('GoogleCloudDialogflowV2MessageAnnotation', 4)
    name = _messages.StringField(5)
    participant = _messages.StringField(6)
    participantRole = _messages.EnumField('ParticipantRoleValueValuesEnum', 7)
    sendTime = _messages.StringField(8)
    sentimentAnalysis = _messages.MessageField('GoogleCloudDialogflowV2SentimentAnalysisResult', 9)