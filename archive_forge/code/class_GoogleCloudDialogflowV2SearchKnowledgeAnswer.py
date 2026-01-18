from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SearchKnowledgeAnswer(_messages.Message):
    """Represents a SearchKnowledge answer.

  Enums:
    AnswerTypeValueValuesEnum: The type of the answer.

  Fields:
    answer: The piece of text from the knowledge base documents that answers
      the search query
    answerRecord: The name of the answer record. Format:
      `projects//locations//answer Records/`
    answerSources: All sources used to generate the answer.
    answerType: The type of the answer.
  """

    class AnswerTypeValueValuesEnum(_messages.Enum):
        """The type of the answer.

    Values:
      ANSWER_TYPE_UNSPECIFIED: The answer has a unspecified type.
      FAQ: The answer is from FAQ documents.
      GENERATIVE: The answer is from generative model.
      INTENT: The answer is from intent matching.
    """
        ANSWER_TYPE_UNSPECIFIED = 0
        FAQ = 1
        GENERATIVE = 2
        INTENT = 3
    answer = _messages.StringField(1)
    answerRecord = _messages.StringField(2)
    answerSources = _messages.MessageField('GoogleCloudDialogflowV2SearchKnowledgeAnswerAnswerSource', 3, repeated=True)
    answerType = _messages.EnumField('AnswerTypeValueValuesEnum', 4)