from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnswerRelevanceValueValuesEnum(_messages.Enum):
    """Optional. Whether or not the suggested answer is relevant. For
    example: * Query: "Can I change my mailing address?" * Suggested document
    says: "Items must be returned/exchanged within 60 days of the purchase
    date." * answer_relevance: AnswerRelevance.IRRELEVANT

    Values:
      ANSWER_RELEVANCE_UNSPECIFIED: Answer relevance unspecified.
      IRRELEVANT: Answer is irrelevant to query.
      RELEVANT: Answer is relevant to query.
    """
    ANSWER_RELEVANCE_UNSPECIFIED = 0
    IRRELEVANT = 1
    RELEVANT = 2