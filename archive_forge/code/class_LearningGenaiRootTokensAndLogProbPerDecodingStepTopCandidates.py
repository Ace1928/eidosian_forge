from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootTokensAndLogProbPerDecodingStepTopCandidates(_messages.Message):
    """Candidates with top log probabilities at each decoding step.

  Fields:
    candidates: Sorted by log probability in descending order.
  """
    candidates = _messages.MessageField('LearningGenaiRootTokensAndLogProbPerDecodingStepCandidate', 1, repeated=True)