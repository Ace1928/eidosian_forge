from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaExplainedAllowPolicy(_messages.Message):
    """Details about how a specific IAM allow policy contributed to the final
  access state.

  Enums:
    AllowAccessStateValueValuesEnum: Required. Indicates whether _this policy_
      provides the specified permission to the specified principal for the
      specified resource. This field does _not_ indicate whether the principal
      actually has the permission for the resource. There might be another
      policy that overrides this policy. To determine whether the principal
      actually has the permission, use the `overall_access_state` field in the
      TroubleshootIamPolicyResponse.
    RelevanceValueValuesEnum: The relevance of this policy to the overall
      access state in the TroubleshootIamPolicyResponse. If the sender of the
      request does not have access to the policy, this field is omitted.

  Fields:
    allowAccessState: Required. Indicates whether _this policy_ provides the
      specified permission to the specified principal for the specified
      resource. This field does _not_ indicate whether the principal actually
      has the permission for the resource. There might be another policy that
      overrides this policy. To determine whether the principal actually has
      the permission, use the `overall_access_state` field in the
      TroubleshootIamPolicyResponse.
    bindingExplanations: Details about how each role binding in the policy
      affects the principal's ability, or inability, to use the permission for
      the resource. The order of the role bindings matches the role binding
      order in the policy. If the sender of the request does not have access
      to the policy, this field is omitted.
    fullResourceName: The full resource name that identifies the resource. For
      example, `//compute.googleapis.com/projects/my-project/zones/us-
      central1-a/instances/my-instance`. If the sender of the request does not
      have access to the policy, this field is omitted. For examples of full
      resource names for Google Cloud services, see
      https://cloud.google.com/iam/help/troubleshooter/full-resource-names.
    policy: The IAM allow policy attached to the resource. If the sender of
      the request does not have access to the policy, this field is empty.
    relevance: The relevance of this policy to the overall access state in the
      TroubleshootIamPolicyResponse. If the sender of the request does not
      have access to the policy, this field is omitted.
  """

    class AllowAccessStateValueValuesEnum(_messages.Enum):
        """Required. Indicates whether _this policy_ provides the specified
    permission to the specified principal for the specified resource. This
    field does _not_ indicate whether the principal actually has the
    permission for the resource. There might be another policy that overrides
    this policy. To determine whether the principal actually has the
    permission, use the `overall_access_state` field in the
    TroubleshootIamPolicyResponse.

    Values:
      ALLOW_ACCESS_STATE_UNSPECIFIED: Not specified.
      ALLOW_ACCESS_STATE_GRANTED: The allow policy gives the principal the
        permission.
      ALLOW_ACCESS_STATE_NOT_GRANTED: The allow policy doesn't give the
        principal the permission.
      ALLOW_ACCESS_STATE_UNKNOWN_CONDITIONAL: The allow policy gives the
        principal the permission if a condition expression evaluate to `true`.
        However, the sender of the request didn't provide enough context for
        Policy Troubleshooter to evaluate the condition expression.
      ALLOW_ACCESS_STATE_UNKNOWN_INFO: The sender of the request doesn't have
        access to all of the allow policies that Policy Troubleshooter needs
        to evaluate the principal's access.
    """
        ALLOW_ACCESS_STATE_UNSPECIFIED = 0
        ALLOW_ACCESS_STATE_GRANTED = 1
        ALLOW_ACCESS_STATE_NOT_GRANTED = 2
        ALLOW_ACCESS_STATE_UNKNOWN_CONDITIONAL = 3
        ALLOW_ACCESS_STATE_UNKNOWN_INFO = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of this policy to the overall access state in the
    TroubleshootIamPolicyResponse. If the sender of the request does not have
    access to the policy, this field is omitted.

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
    allowAccessState = _messages.EnumField('AllowAccessStateValueValuesEnum', 1)
    bindingExplanations = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaAllowBindingExplanation', 2, repeated=True)
    fullResourceName = _messages.StringField(3)
    policy = _messages.MessageField('GoogleIamV1Policy', 4)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 5)