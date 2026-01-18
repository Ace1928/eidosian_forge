from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootSimilarityTakedownResult(_messages.Message):
    """A LearningGenaiRootSimilarityTakedownResult object.

  Fields:
    allowed: False when query or response should be taken down by any of the
      takedown rules, true otherwise.
    scoredPhrases: List of similar phrases with score. Set only if
      allowed=false.
  """
    allowed = _messages.BooleanField(1)
    scoredPhrases = _messages.MessageField('LearningGenaiRootScoredSimilarityTakedownPhrase', 2, repeated=True)