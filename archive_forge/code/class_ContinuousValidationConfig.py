from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ContinuousValidationConfig(_messages.Message):
    """A user config for specifying the continuous validation (CV) settings for
  various policies. There is at most one config per project (a singleton
  resource).

  Fields:
    enforcementPolicyConfig: The continuous validation config for enforcement
      policy.
    name: Output only. The resource name, in the format
      `projects/*/continuousValidationConfig`. There is at most one config per
      project.
    updateTime: Output only. Time when the config was last updated.
  """
    enforcementPolicyConfig = _messages.MessageField('EnforcementPolicyConfig', 1)
    name = _messages.StringField(2)
    updateTime = _messages.StringField(3)