from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2Conversation(_messages.Message):
    """Represents a conversation. A conversation is an interaction between an
  agent, including live agents and Dialogflow agents, and a support customer.
  Conversations can include phone calls and text-based chat sessions.

  Enums:
    ConversationStageValueValuesEnum: The stage of a conversation. It
      indicates whether the virtual agent or a human agent is handling the
      conversation. If the conversation is created with the conversation
      profile that has Dialogflow config set, defaults to
      ConversationStage.VIRTUAL_AGENT_STAGE; Otherwise, defaults to
      ConversationStage.HUMAN_ASSIST_STAGE. If the conversation is created
      with the conversation profile that has Dialogflow config set but
      explicitly sets conversation_stage to
      ConversationStage.HUMAN_ASSIST_STAGE, it skips
      ConversationStage.VIRTUAL_AGENT_STAGE stage and directly goes to
      ConversationStage.HUMAN_ASSIST_STAGE.
    LifecycleStateValueValuesEnum: Output only. The current state of the
      Conversation.

  Fields:
    conversationProfile: Required. The Conversation Profile to be used to
      configure this Conversation. This field cannot be updated. Format:
      `projects//locations//conversationProfiles/`.
    conversationStage: The stage of a conversation. It indicates whether the
      virtual agent or a human agent is handling the conversation. If the
      conversation is created with the conversation profile that has
      Dialogflow config set, defaults to
      ConversationStage.VIRTUAL_AGENT_STAGE; Otherwise, defaults to
      ConversationStage.HUMAN_ASSIST_STAGE. If the conversation is created
      with the conversation profile that has Dialogflow config set but
      explicitly sets conversation_stage to
      ConversationStage.HUMAN_ASSIST_STAGE, it skips
      ConversationStage.VIRTUAL_AGENT_STAGE stage and directly goes to
      ConversationStage.HUMAN_ASSIST_STAGE.
    endTime: Output only. The time the conversation was finished.
    lifecycleState: Output only. The current state of the Conversation.
    name: Output only. The unique identifier of this conversation. Format:
      `projects//locations//conversations/`.
    phoneNumber: Output only. It will not be empty if the conversation is to
      be connected over telephony.
    startTime: Output only. The time the conversation was started.
  """

    class ConversationStageValueValuesEnum(_messages.Enum):
        """The stage of a conversation. It indicates whether the virtual agent or
    a human agent is handling the conversation. If the conversation is created
    with the conversation profile that has Dialogflow config set, defaults to
    ConversationStage.VIRTUAL_AGENT_STAGE; Otherwise, defaults to
    ConversationStage.HUMAN_ASSIST_STAGE. If the conversation is created with
    the conversation profile that has Dialogflow config set but explicitly
    sets conversation_stage to ConversationStage.HUMAN_ASSIST_STAGE, it skips
    ConversationStage.VIRTUAL_AGENT_STAGE stage and directly goes to
    ConversationStage.HUMAN_ASSIST_STAGE.

    Values:
      CONVERSATION_STAGE_UNSPECIFIED: Unknown. Should never be used after a
        conversation is successfully created.
      VIRTUAL_AGENT_STAGE: The conversation should return virtual agent
        responses into the conversation.
      HUMAN_ASSIST_STAGE: The conversation should not provide responses, just
        listen and provide suggestions.
    """
        CONVERSATION_STAGE_UNSPECIFIED = 0
        VIRTUAL_AGENT_STAGE = 1
        HUMAN_ASSIST_STAGE = 2

    class LifecycleStateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the Conversation.

    Values:
      LIFECYCLE_STATE_UNSPECIFIED: Unknown.
      IN_PROGRESS: Conversation is currently open for media analysis.
      COMPLETED: Conversation has been completed.
    """
        LIFECYCLE_STATE_UNSPECIFIED = 0
        IN_PROGRESS = 1
        COMPLETED = 2
    conversationProfile = _messages.StringField(1)
    conversationStage = _messages.EnumField('ConversationStageValueValuesEnum', 2)
    endTime = _messages.StringField(3)
    lifecycleState = _messages.EnumField('LifecycleStateValueValuesEnum', 4)
    name = _messages.StringField(5)
    phoneNumber = _messages.MessageField('GoogleCloudDialogflowV2ConversationPhoneNumber', 6)
    startTime = _messages.StringField(7)