from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRulePreconfiguredWafConfig(_messages.Message):
    """A SecurityPolicyRulePreconfiguredWafConfig object.

  Fields:
    exclusions: A list of exclusions to apply during preconfigured WAF
      evaluation.
  """
    exclusions = _messages.MessageField('SecurityPolicyRulePreconfiguredWafConfigExclusion', 1, repeated=True)