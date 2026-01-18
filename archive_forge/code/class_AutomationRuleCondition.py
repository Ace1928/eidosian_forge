from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutomationRuleCondition(_messages.Message):
    """`AutomationRuleCondition` contains conditions relevant to an
  `Automation` rule.

  Fields:
    targetsPresentCondition: Optional. Details around targets enumerated in
      the rule.
  """
    targetsPresentCondition = _messages.MessageField('TargetsPresentCondition', 1)