from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig(_messages.Message):
    """Configuration options for L7 DDoS detection. This field is only
  supported in Global Security Policies of type CLOUD_ARMOR.

  Enums:
    RuleVisibilityValueValuesEnum: Rule visibility can be one of the
      following: STANDARD - opaque rules. (default) PREMIUM - transparent
      rules. This field is only supported in Global Security Policies of type
      CLOUD_ARMOR.

  Fields:
    enable: If set to true, enables CAAP for L7 DDoS detection. This field is
      only supported in Global Security Policies of type CLOUD_ARMOR.
    ruleVisibility: Rule visibility can be one of the following: STANDARD -
      opaque rules. (default) PREMIUM - transparent rules. This field is only
      supported in Global Security Policies of type CLOUD_ARMOR.
    thresholdConfigs: Configuration options for layer7 adaptive protection for
      various customizable thresholds.
  """

    class RuleVisibilityValueValuesEnum(_messages.Enum):
        """Rule visibility can be one of the following: STANDARD - opaque rules.
    (default) PREMIUM - transparent rules. This field is only supported in
    Global Security Policies of type CLOUD_ARMOR.

    Values:
      PREMIUM: <no description>
      STANDARD: <no description>
    """
        PREMIUM = 0
        STANDARD = 1
    enable = _messages.BooleanField(1)
    ruleVisibility = _messages.EnumField('RuleVisibilityValueValuesEnum', 2)
    thresholdConfigs = _messages.MessageField('SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfigThresholdConfig', 3, repeated=True)