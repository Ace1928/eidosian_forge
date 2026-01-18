from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaExplainedPABRule(_messages.Message):
    """Details about how a Principal Access Boundary rule contributes to the
  explanation, with annotations to indicate how the rule contributes to the
  overall access state.

  Enums:
    CombinedResourceInclusionStateValueValuesEnum: Output only. Indicates
      whether any resource of the rule is the specified resource or includes
      the specified resource.
    EffectValueValuesEnum: Required. The effect of the rule which describes
      the access relationship.
    RelevanceValueValuesEnum: The relevance of this rule to the overall access
      state.
    RuleAccessStateValueValuesEnum: Output only. Indicates whether the rule
      allows access to the specified resource.

  Fields:
    combinedResourceInclusionState: Output only. Indicates whether any
      resource of the rule is the specified resource or includes the specified
      resource.
    effect: Required. The effect of the rule which describes the access
      relationship.
    explainedResources: List of resources that were explained to check the
      principal's access to specified resource, with annotations to indicate
      how each resource contributes to the overall access state.
    relevance: The relevance of this rule to the overall access state.
    ruleAccessState: Output only. Indicates whether the rule allows access to
      the specified resource.
  """

    class CombinedResourceInclusionStateValueValuesEnum(_messages.Enum):
        """Output only. Indicates whether any resource of the rule is the
    specified resource or includes the specified resource.

    Values:
      RESOURCE_INCLUSION_STATE_UNSPECIFIED: An error occurred when checking
        whether the resource includes the specified resource.
      RESOURCE_INCLUSION_STATE_INCLUDED: The resource includes the specified
        resource.
      RESOURCE_INCLUSION_STATE_NOT_INCLUDED: The resource doesn't include the
        specified resource.
      RESOURCE_INCLUSION_STATE_UNKNOWN_INFO: The sender of the request does
        not have access to the relevant data to check whether the resource
        includes the specified resource.
      RESOURCE_INCLUSION_STATE_UNKNOWN_UNSUPPORTED: The resource is of an
        unsupported type, such as non-CRM resources.
    """
        RESOURCE_INCLUSION_STATE_UNSPECIFIED = 0
        RESOURCE_INCLUSION_STATE_INCLUDED = 1
        RESOURCE_INCLUSION_STATE_NOT_INCLUDED = 2
        RESOURCE_INCLUSION_STATE_UNKNOWN_INFO = 3
        RESOURCE_INCLUSION_STATE_UNKNOWN_UNSUPPORTED = 4

    class EffectValueValuesEnum(_messages.Enum):
        """Required. The effect of the rule which describes the access
    relationship.

    Values:
      EFFECT_UNSPECIFIED: Effect unspecified.
      ALLOW: Allows access to the resources in this rule.
    """
        EFFECT_UNSPECIFIED = 0
        ALLOW = 1

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of this rule to the overall access state.

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

    class RuleAccessStateValueValuesEnum(_messages.Enum):
        """Output only. Indicates whether the rule allows access to the specified
    resource.

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
    combinedResourceInclusionState = _messages.EnumField('CombinedResourceInclusionStateValueValuesEnum', 1)
    effect = _messages.EnumField('EffectValueValuesEnum', 2)
    explainedResources = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaExplainedPABRuleExplainedResource', 3, repeated=True)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 4)
    ruleAccessState = _messages.EnumField('RuleAccessStateValueValuesEnum', 5)