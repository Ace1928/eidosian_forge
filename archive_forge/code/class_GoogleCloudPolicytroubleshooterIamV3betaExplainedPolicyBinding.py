from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaExplainedPolicyBinding(_messages.Message):
    """Details about how a policy binding contributes to the policy
  explanation, with annotations to indicate how the policy binding contributes
  to the overall access state.

  Enums:
    PolicyBindingStateValueValuesEnum: Output only. Indicates whether the
      policy binding takes effect.
    RelevanceValueValuesEnum: The relevance of this policy binding to the
      overall access state.

  Fields:
    conditionExplanation: Optional. Explanation of the condition in the policy
      binding. If the policy binding doesn't have a condition, this field is
      omitted.
    policyBinding: The policy binding that is explained.
    policyBindingState: Output only. Indicates whether the policy binding
      takes effect.
    relevance: The relevance of this policy binding to the overall access
      state.
  """

    class PolicyBindingStateValueValuesEnum(_messages.Enum):
        """Output only. Indicates whether the policy binding takes effect.

    Values:
      POLICY_BINDING_STATE_UNSPECIFIED: An error occurred when checking
        whether the policy binding is enforced.
      POLICY_BINDING_STATE_ENFORCED: The policy binding is enforced.
      POLICY_BINDING_STATE_NOT_ENFORCED: The policy binding is not enforced.
    """
        POLICY_BINDING_STATE_UNSPECIFIED = 0
        POLICY_BINDING_STATE_ENFORCED = 1
        POLICY_BINDING_STATE_NOT_ENFORCED = 2

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of this policy binding to the overall access state.

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
    conditionExplanation = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3betaConditionExplanation', 1)
    policyBinding = _messages.MessageField('GoogleIamV3PolicyBinding', 2)
    policyBindingState = _messages.EnumField('PolicyBindingStateValueValuesEnum', 3)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 4)