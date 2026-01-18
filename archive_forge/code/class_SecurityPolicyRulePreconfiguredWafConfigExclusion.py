from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRulePreconfiguredWafConfigExclusion(_messages.Message):
    """A SecurityPolicyRulePreconfiguredWafConfigExclusion object.

  Fields:
    requestCookiesToExclude: A list of request cookie names whose value will
      be excluded from inspection during preconfigured WAF evaluation.
    requestHeadersToExclude: A list of request header names whose value will
      be excluded from inspection during preconfigured WAF evaluation.
    requestQueryParamsToExclude: A list of request query parameter names whose
      value will be excluded from inspection during preconfigured WAF
      evaluation. Note that the parameter can be in the query string or in the
      POST body.
    requestUrisToExclude: A list of request URIs from the request line to be
      excluded from inspection during preconfigured WAF evaluation. When
      specifying this field, the query or fragment part should be excluded.
    targetRuleIds: A list of target rule IDs under the WAF rule set to apply
      the preconfigured WAF exclusion. If omitted, it refers to all the rule
      IDs under the WAF rule set.
    targetRuleSet: Target WAF rule set to apply the preconfigured WAF
      exclusion.
  """
    requestCookiesToExclude = _messages.MessageField('SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams', 1, repeated=True)
    requestHeadersToExclude = _messages.MessageField('SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams', 2, repeated=True)
    requestQueryParamsToExclude = _messages.MessageField('SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams', 3, repeated=True)
    requestUrisToExclude = _messages.MessageField('SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams', 4, repeated=True)
    targetRuleIds = _messages.StringField(5, repeated=True)
    targetRuleSet = _messages.StringField(6)