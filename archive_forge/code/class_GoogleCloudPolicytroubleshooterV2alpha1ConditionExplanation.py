from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterV2alpha1ConditionExplanation(_messages.Message):
    """Condition Explanation

  Fields:
    evaluationStates: List of evaluated states of non boolean expression in
      the condition
    value: Value of the condition
  """
    evaluationStates = _messages.MessageField('GoogleCloudPolicytroubleshooterV2alpha1ConditionExplanationEvaluationState', 1, repeated=True)
    value = _messages.MessageField('extra_types.JsonValue', 2)