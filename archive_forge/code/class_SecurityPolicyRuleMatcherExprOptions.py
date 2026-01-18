from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleMatcherExprOptions(_messages.Message):
    """A SecurityPolicyRuleMatcherExprOptions object.

  Fields:
    recaptchaOptions: reCAPTCHA configuration options to be applied for the
      rule. If the rule does not evaluate reCAPTCHA tokens, this field has no
      effect.
  """
    recaptchaOptions = _messages.MessageField('SecurityPolicyRuleMatcherExprOptionsRecaptchaOptions', 1)