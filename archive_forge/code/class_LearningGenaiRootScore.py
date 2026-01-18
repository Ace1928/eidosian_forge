from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootScore(_messages.Message):
    """A LearningGenaiRootScore object.

  Fields:
    calculationType: A LearningGenaiRootCalculationType attribute.
    internalMetadata: The internal_metadata is intended to be used by internal
      processors and will be cleared before returns.
    thresholdType: A LearningGenaiRootThresholdType attribute.
    tokensAndLogprobPerDecodingStep: Top candidate tokens and log
      probabilities at each decoding step.
    value: A number attribute.
  """
    calculationType = _messages.MessageField('LearningGenaiRootCalculationType', 1)
    internalMetadata = _messages.MessageField('LearningGenaiRootInternalMetadata', 2)
    thresholdType = _messages.MessageField('LearningGenaiRootThresholdType', 3)
    tokensAndLogprobPerDecodingStep = _messages.MessageField('LearningGenaiRootTokensAndLogProbPerDecodingStep', 4)
    value = _messages.FloatField(5)