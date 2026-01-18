from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeOrgPoliciesResponse(_messages.Message):
    """The response message for AssetService.AnalyzeOrgPolicies.

  Fields:
    constraint: The definition of the constraint in the request.
    nextPageToken: The page token to fetch the next page for
      AnalyzeOrgPoliciesResponse.org_policy_results.
    orgPolicyResults: The organization policies under the
      AnalyzeOrgPoliciesRequest.scope with the
      AnalyzeOrgPoliciesRequest.constraint.
  """
    constraint = _messages.MessageField('AnalyzerOrgPolicyConstraint', 1)
    nextPageToken = _messages.StringField(2)
    orgPolicyResults = _messages.MessageField('OrgPolicyResult', 3, repeated=True)