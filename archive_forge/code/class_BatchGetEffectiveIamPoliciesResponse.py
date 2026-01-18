from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchGetEffectiveIamPoliciesResponse(_messages.Message):
    """A response message for AssetService.BatchGetEffectiveIamPolicies.

  Fields:
    policyResults: The effective policies for a batch of resources. Note that
      the results order is the same as the order of
      BatchGetEffectiveIamPoliciesRequest.names. When a resource does not have
      any effective IAM policies, its corresponding policy_result will contain
      empty EffectiveIamPolicy.policies.
  """
    policyResults = _messages.MessageField('EffectiveIamPolicy', 1, repeated=True)