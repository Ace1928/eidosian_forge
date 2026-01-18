from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootTokensAndLogProbPerDecodingStep(_messages.Message):
    """Results of RandomSamplingParams::top_k_logprob_per_decoding_step.

  Fields:
    chosenCandidates: Length = total number of decoding steps. The chosen
      candidates may or may not be in top_candidates.
    topCandidates: Length = total number of decoding steps.
  """
    chosenCandidates = _messages.MessageField('LearningGenaiRootTokensAndLogProbPerDecodingStepCandidate', 1, repeated=True)
    topCandidates = _messages.MessageField('LearningGenaiRootTokensAndLogProbPerDecodingStepTopCandidates', 2, repeated=True)