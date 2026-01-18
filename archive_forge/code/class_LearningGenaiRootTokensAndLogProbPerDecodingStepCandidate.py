from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootTokensAndLogProbPerDecodingStepCandidate(_messages.Message):
    """A candidate at a decoding step.

  Fields:
    logProbability: The candidate's log probability.
    token: The candidate's token value.
  """
    logProbability = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    token = _messages.StringField(2)