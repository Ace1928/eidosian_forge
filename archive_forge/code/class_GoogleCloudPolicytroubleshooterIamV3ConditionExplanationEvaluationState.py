from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3ConditionExplanationEvaluationState(_messages.Message):
    """Evaluated state of a condition expression.

  Fields:
    end: End position of an expression in the condition, by character, end
      included, for example: the end position of the first part of `a==b ||
      c==d` would be 4.
    errors: Any errors that prevented complete evaluation of the condition
      expression.
    start: Start position of an expression in the condition, by character.
    value: Value of this expression.
  """
    end = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    errors = _messages.MessageField('GoogleRpcStatus', 2, repeated=True)
    start = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    value = _messages.MessageField('extra_types.JsonValue', 4)