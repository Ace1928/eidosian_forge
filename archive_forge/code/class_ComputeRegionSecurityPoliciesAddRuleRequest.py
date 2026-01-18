from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionSecurityPoliciesAddRuleRequest(_messages.Message):
    """A ComputeRegionSecurityPoliciesAddRuleRequest object.

  Fields:
    project: Project ID for this request.
    region: Name of the region scoping this request.
    securityPolicy: Name of the security policy to update.
    securityPolicyRule: A SecurityPolicyRule resource to be passed as the
      request body.
    validateOnly: If true, the request will not be committed.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    securityPolicy = _messages.StringField(3, required=True)
    securityPolicyRule = _messages.MessageField('SecurityPolicyRule', 4)
    validateOnly = _messages.BooleanField(5)