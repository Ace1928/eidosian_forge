from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckResult(_messages.Message):
    """Result of evaluating one check.

  Fields:
    allowlistResult: If the image was exempted by an allow_pattern in the
      check, contains the pattern that the image name matched.
    displayName: The name of the check.
    evaluationResult: If a check was evaluated, contains the result of the
      check.
    explanation: Explanation of this check result.
    index: The index of the check.
    type: The type of the check.
  """
    allowlistResult = _messages.MessageField('AllowlistResult', 1)
    displayName = _messages.StringField(2)
    evaluationResult = _messages.MessageField('EvaluationResult', 3)
    explanation = _messages.StringField(4)
    index = _messages.IntegerField(5)
    type = _messages.StringField(6)