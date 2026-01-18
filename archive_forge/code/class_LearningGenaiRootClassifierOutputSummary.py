from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootClassifierOutputSummary(_messages.Message):
    """A LearningGenaiRootClassifierOutputSummary object.

  Fields:
    metrics: A LearningGenaiRootMetricOutput attribute.
    ruleOutput: Output of the first matching rule.
    ruleOutputs: outputs of all matching rule.
  """
    metrics = _messages.MessageField('LearningGenaiRootMetricOutput', 1, repeated=True)
    ruleOutput = _messages.MessageField('LearningGenaiRootRuleOutput', 2)
    ruleOutputs = _messages.MessageField('LearningGenaiRootRuleOutput', 3, repeated=True)