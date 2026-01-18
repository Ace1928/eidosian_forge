from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectivePolicy(_messages.Message):
    """Effective Policy is a singleton read-only resource modeling the
  collapsed policies and metadata effective at a particular point in the
  hierarchy.

  Fields:
    enableRuleMetadata: Output only. Metadata about enable rules in the same
      order as the `enable_rules` objects.
    enableRules: Output only. Aggregated `EnableRule` objects grouped by any
      associated conditions. Conditions are not supported in `alpha` and there
      will be exactly one rule present.
    name: Output only. The name of the effective policy. Format:
      `projects/100/effectivePolicy`, `folders/101/effectivePolicy`,
      `organizations/102/effectivePolicy`.
    updateTime: Output only. The time the policy was last updated.
  """
    enableRuleMetadata = _messages.MessageField('RuleSource', 1, repeated=True)
    enableRules = _messages.MessageField('GoogleApiServiceusageV2alphaEnableRule', 2, repeated=True)
    name = _messages.StringField(3)
    updateTime = _messages.StringField(4)