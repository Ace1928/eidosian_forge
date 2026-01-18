from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2AnswerFeedback(_messages.Message):
    """Represents feedback the customer has about the quality & correctness of
  a certain answer in a conversation.

  Enums:
    CorrectnessLevelValueValuesEnum: The correctness level of the specific
      answer.

  Fields:
    agentAssistantDetailFeedback: Detail feedback of agent assist suggestions.
    clickTime: Time when the answer/item was clicked.
    clicked: Indicates whether the answer/item was clicked by the human agent
      or not. Default to false. For knowledge search and knowledge assist, the
      answer record is considered to be clicked if the answer was copied or
      any URI was clicked.
    correctnessLevel: The correctness level of the specific answer.
    displayTime: Time when the answer/item was displayed.
    displayed: Indicates whether the answer/item was displayed to the human
      agent in the agent desktop UI. Default to false.
  """

    class CorrectnessLevelValueValuesEnum(_messages.Enum):
        """The correctness level of the specific answer.

    Values:
      CORRECTNESS_LEVEL_UNSPECIFIED: Correctness level unspecified.
      NOT_CORRECT: Answer is totally wrong.
      PARTIALLY_CORRECT: Answer is partially correct.
      FULLY_CORRECT: Answer is fully correct.
    """
        CORRECTNESS_LEVEL_UNSPECIFIED = 0
        NOT_CORRECT = 1
        PARTIALLY_CORRECT = 2
        FULLY_CORRECT = 3
    agentAssistantDetailFeedback = _messages.MessageField('GoogleCloudDialogflowV2AgentAssistantFeedback', 1)
    clickTime = _messages.StringField(2)
    clicked = _messages.BooleanField(3)
    correctnessLevel = _messages.EnumField('CorrectnessLevelValueValuesEnum', 4)
    displayTime = _messages.StringField(5)
    displayed = _messages.BooleanField(6)