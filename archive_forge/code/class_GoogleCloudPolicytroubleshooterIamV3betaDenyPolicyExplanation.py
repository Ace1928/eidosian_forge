from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaDenyPolicyExplanation(_messages.Message):
    """Details about how the relevant IAM deny policies affect the final access
  state.

  Enums:
    DenyAccessStateValueValuesEnum: Indicates whether the principal is denied
      the specified permission for the specified resource, based on evaluating
      all applicable IAM deny policies.
    RelevanceValueValuesEnum: The relevance of the deny policy result to the
      overall access state.

  Fields:
    denyAccessState: Indicates whether the principal is denied the specified
      permission for the specified resource, based on evaluating all
      applicable IAM deny policies.
    explainedResources: List of resources with IAM deny policies that were
      evaluated to check the principal's denied permissions, with annotations
      to indicate how each policy contributed to the final result. The list of
      resources includes the policy for the resource itself, as well as
      policies that are inherited from higher levels of the resource
      hierarchy, including the organization, the folder, and the project. The
      order of the resources starts from the resource and climbs up the
      resource hierarchy. To learn more about the resource hierarchy, see
      https://cloud.google.com/iam/help/resource-hierarchy.
    permissionDeniable: Indicates whether the permission to troubleshoot is
      supported in deny policies.
    relevance: The relevance of the deny policy result to the overall access
      state.
  """

    class DenyAccessStateValueValuesEnum(_messages.Enum):
        """Indicates whether the principal is denied the specified permission for
    the specified resource, based on evaluating all applicable IAM deny
    policies.

    Values:
      DENY_ACCESS_STATE_UNSPECIFIED: Not specified.
      DENY_ACCESS_STATE_DENIED: The deny policy denies the principal the
        permission.
      DENY_ACCESS_STATE_NOT_DENIED: The deny policy doesn't deny the principal
        the permission.
      DENY_ACCESS_STATE_UNKNOWN_CONDITIONAL: The deny policy denies the
        principal the permission if a condition expression evaluates to
        `true`. However, the sender of the request didn't provide enough
        context for Policy Troubleshooter to evaluate the condition
        expression.
      DENY_ACCESS_STATE_UNKNOWN_INFO: The sender of the request does not have
        access to all of the deny policies that Policy Troubleshooter needs to
        evaluate the principal's access.
    """
        DENY_ACCESS_STATE_UNSPECIFIED = 0
        DENY_ACCESS_STATE_DENIED = 1
        DENY_ACCESS_STATE_NOT_DENIED = 2
        DENY_ACCESS_STATE_UNKNOWN_CONDITIONAL = 3
        DENY_ACCESS_STATE_UNKNOWN_INFO = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of the deny policy result to the overall access state.

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
    denyAccessState = _messages.EnumField('DenyAccessStateValueValuesEnum', 1)
    explainedResources = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3betaExplainedDenyResource', 2, repeated=True)
    permissionDeniable = _messages.BooleanField(3)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 4)