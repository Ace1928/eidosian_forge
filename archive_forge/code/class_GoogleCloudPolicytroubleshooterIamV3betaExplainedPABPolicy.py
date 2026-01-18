from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaExplainedPABPolicy(_messages.Message):
    """Details about how a Principal Access Boundary policy contributes to the
  explanation, with annotations to indicate how the policy contributes to the
  overall access state.

  Enums:
    PolicyAccessStateValueValuesEnum: Output only. Indicates whether the
      policy allows access to the specified resource.
    RelevanceValueValuesEnum: The relevance of this policy to the overall
      access state.

  Fields:
    explainedRules: List of Principal Access Boundary rules that were
      explained to check the principal's access to specified resource, with
      annotations to indicate how each rule contributes to the overall access
      state.
    policy: The policy that is explained.
    policyAccessState: Output only. Indicates whether the policy allows access
      to the specified resource.
    policyVersion: Output only. Explanation of the Principal Access Boundary
      policy's version.
    relevance: The relevance of this policy to the overall access state.
  """

    class PolicyAccessStateValueValuesEnum(_messages.Enum):
        """Output only. Indicates whether the policy allows access to the
    specified resource.

    Values:
      PAB_ACCESS_STATE_UNSPECIFIED: Not specified.
      PAB_ACCESS_STATE_ALLOWED: The PAB component allows the principal's
        access to the specified resource.
      PAB_ACCESS_STATE_NOT_ALLOWED: The PAB component doesn't allow the
        principal's access to the specified resource.
      PAB_ACCESS_STATE_NOT_ENFORCED: The PAB component is not enforced on the
        principal, or the specified resource. This state refers to 2 specific
        scenarios: - The service that the specified resource belongs to is not
        enforced by PAB at the policy version. - The binding doesn't apply to
        the principal, hence the policy is not enforced as a result.
      PAB_ACCESS_STATE_UNKNOWN_INFO: The sender of the request does not have
        access to the PAB component, or the relevant data to explain the PAB
        component.
    """
        PAB_ACCESS_STATE_UNSPECIFIED = 0
        PAB_ACCESS_STATE_ALLOWED = 1
        PAB_ACCESS_STATE_NOT_ALLOWED = 2
        PAB_ACCESS_STATE_NOT_ENFORCED = 3
        PAB_ACCESS_STATE_UNKNOWN_INFO = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of this policy to the overall access state.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Not specified.
      HEURISTIC_RELEVANCE_NORMAL: The data point has a limited effect on the
        result. Changing the data point is unlikely to affect the overall
        determination.
      HEURISTIC_RELEVANCE_HIGH: The data point has a strong effect on the
        result. Changing the data point is likely to affect the overall
        determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        HEURISTIC_RELEVANCE_NORMAL = 1
        HEURISTIC_RELEVANCE_HIGH = 2
    explainedRules = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3betaExplainedPABRule', 1, repeated=True)
    policy = _messages.MessageField('GoogleIamV3PrincipalAccessBoundaryPolicy', 2)
    policyAccessState = _messages.EnumField('PolicyAccessStateValueValuesEnum', 3)
    policyVersion = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3betaExplainedPABPolicyVersion', 4)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 5)