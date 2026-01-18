from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaConditionExplanation(_messages.Message):
    """Explanation for how a condition affects a principal's access

  Fields:
    errors: Any errors that prevented complete evaluation of the condition
      expression.
    evaluationStates: The value of each statement of the condition expression.
      The value can be `true`, `false`, or `null`. The value is `null` if the
      statement can't be evaluated.
    value: Value of the condition.
  """
    errors = _messages.MessageField('GoogleRpcStatus', 1, repeated=True)
    evaluationStates = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaConditionExplanationEvaluationState', 2, repeated=True)
    value = _messages.MessageField('extra_types.JsonValue', 3)