from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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