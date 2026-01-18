from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckSetResult(_messages.Message):
    """Result of evaluating one check set.

  Fields:
    allowlistResult: If the image was exempted by an allow_pattern in the
      check set, contains the pattern that the image name matched.
    checkResults: If checks were evaluated, contains the results of evaluating
      each check.
    displayName: The name of the check set.
    explanation: Explanation of this check set result. Only populated if no
      checks were evaluated.
    index: The index of the check set.
    scope: The scope of the check set.
  """
    allowlistResult = _messages.MessageField('AllowlistResult', 1)
    checkResults = _messages.MessageField('CheckResults', 2)
    displayName = _messages.StringField(3)
    explanation = _messages.StringField(4)
    index = _messages.IntegerField(5)
    scope = _messages.MessageField('Scope', 6)