from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyAdaptiveProtectionConfig(_messages.Message):
    """Configuration options for Cloud Armor Adaptive Protection (CAAP).

  Fields:
    autoDeployConfig: A SecurityPolicyAdaptiveProtectionConfigAutoDeployConfig
      attribute.
    layer7DdosDefenseConfig: If set to true, enables Cloud Armor Machine
      Learning.
  """
    autoDeployConfig = _messages.MessageField('SecurityPolicyAdaptiveProtectionConfigAutoDeployConfig', 1)
    layer7DdosDefenseConfig = _messages.MessageField('SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig', 2)