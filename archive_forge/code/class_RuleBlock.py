from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuleBlock(_messages.Message):
    """RuleBlock holds the action and rule definitions of an authorization
  policy.

  Enums:
    ActionValueValuesEnum: Action type of this policy.

  Fields:
    action: Action type of this policy.
    rules: Rules that must be evaluated for this policy action.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Action type of this policy.

    Values:
      ACTION_UNSPECIFIED: Policy rules with no action type.
      ALLOW: Maps to allow policy rules.
      DENY: Maps to deny policy rules.
    """
        ACTION_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    rules = _messages.MessageField('Rule', 2, repeated=True)