from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaAllowBindingExplanation(_messages.Message):
    """Details about how a role binding in an allow policy affects a
  principal's ability to use a permission.

  Enums:
    AllowAccessStateValueValuesEnum: Required. Indicates whether _this role
      binding_ gives the specified permission to the specified principal on
      the specified resource. This field does _not_ indicate whether the
      principal actually has the permission on the resource. There might be
      another role binding that overrides this role binding. To determine
      whether the principal actually has the permission, use the
      `overall_access_state` field in the TroubleshootIamPolicyResponse.
    RelevanceValueValuesEnum: The relevance of this role binding to the
      overall determination for the entire policy.
    RolePermissionValueValuesEnum: Indicates whether the role granted by this
      role binding contains the specified permission.
    RolePermissionRelevanceValueValuesEnum: The relevance of the permission's
      existence, or nonexistence, in the role to the overall determination for
      the entire policy.

  Messages:
    MembershipsValue: Indicates whether each role binding includes the
      principal specified in the request, either directly or indirectly. Each
      key identifies a principal in the role binding, and each value indicates
      whether the principal in the role binding includes the principal in the
      request. For example, suppose that a role binding includes the following
      principals: * `user:alice@example.com` * `group:product-eng@example.com`
      You want to troubleshoot access for `user:bob@example.com`. This user is
      a member of the group `group:product-eng@example.com`. For the first
      principal in the role binding, the key is `user:alice@example.com`, and
      the `membership` field in the value is set to `NOT_INCLUDED`. For the
      second principal in the role binding, the key is `group:product-
      eng@example.com`, and the `membership` field in the value is set to
      `INCLUDED`.

  Fields:
    allowAccessState: Required. Indicates whether _this role binding_ gives
      the specified permission to the specified principal on the specified
      resource. This field does _not_ indicate whether the principal actually
      has the permission on the resource. There might be another role binding
      that overrides this role binding. To determine whether the principal
      actually has the permission, use the `overall_access_state` field in the
      TroubleshootIamPolicyResponse.
    combinedMembership: The combined result of all memberships. Indicates if
      the principal is included in any role binding, either directly or
      indirectly.
    condition: A condition expression that specifies when the role binding
      grants access. To learn about IAM Conditions, see
      https://cloud.google.com/iam/help/conditions/overview.
    conditionExplanation: Condition evaluation state for this role binding.
    memberships: Indicates whether each role binding includes the principal
      specified in the request, either directly or indirectly. Each key
      identifies a principal in the role binding, and each value indicates
      whether the principal in the role binding includes the principal in the
      request. For example, suppose that a role binding includes the following
      principals: * `user:alice@example.com` * `group:product-eng@example.com`
      You want to troubleshoot access for `user:bob@example.com`. This user is
      a member of the group `group:product-eng@example.com`. For the first
      principal in the role binding, the key is `user:alice@example.com`, and
      the `membership` field in the value is set to `NOT_INCLUDED`. For the
      second principal in the role binding, the key is `group:product-
      eng@example.com`, and the `membership` field in the value is set to
      `INCLUDED`.
    relevance: The relevance of this role binding to the overall determination
      for the entire policy.
    role: The role that this role binding grants. For example,
      `roles/compute.admin`. For a complete list of predefined IAM roles, as
      well as the permissions in each role, see
      https://cloud.google.com/iam/help/roles/reference.
    rolePermission: Indicates whether the role granted by this role binding
      contains the specified permission.
    rolePermissionRelevance: The relevance of the permission's existence, or
      nonexistence, in the role to the overall determination for the entire
      policy.
  """

    class AllowAccessStateValueValuesEnum(_messages.Enum):
        """Required. Indicates whether _this role binding_ gives the specified
    permission to the specified principal on the specified resource. This
    field does _not_ indicate whether the principal actually has the
    permission on the resource. There might be another role binding that
    overrides this role binding. To determine whether the principal actually
    has the permission, use the `overall_access_state` field in the
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
        """The relevance of this role binding to the overall determination for
    the entire policy.

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

    class RolePermissionRelevanceValueValuesEnum(_messages.Enum):
        """The relevance of the permission's existence, or nonexistence, in the
    role to the overall determination for the entire policy.

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

    class RolePermissionValueValuesEnum(_messages.Enum):
        """Indicates whether the role granted by this role binding contains the
    specified permission.

    Values:
      ROLE_PERMISSION_INCLUSION_STATE_UNSPECIFIED: Not specified.
      ROLE_PERMISSION_INCLUDED: The permission is included in the role.
      ROLE_PERMISSION_NOT_INCLUDED: The permission is not included in the
        role.
      ROLE_PERMISSION_UNKNOWN_INFO: The sender of the request is not allowed
        to access the role definition.
    """
        ROLE_PERMISSION_INCLUSION_STATE_UNSPECIFIED = 0
        ROLE_PERMISSION_INCLUDED = 1
        ROLE_PERMISSION_NOT_INCLUDED = 2
        ROLE_PERMISSION_UNKNOWN_INFO = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MembershipsValue(_messages.Message):
        """Indicates whether each role binding includes the principal specified
    in the request, either directly or indirectly. Each key identifies a
    principal in the role binding, and each value indicates whether the
    principal in the role binding includes the principal in the request. For
    example, suppose that a role binding includes the following principals: *
    `user:alice@example.com` * `group:product-eng@example.com` You want to
    troubleshoot access for `user:bob@example.com`. This user is a member of
    the group `group:product-eng@example.com`. For the first principal in the
    role binding, the key is `user:alice@example.com`, and the `membership`
    field in the value is set to `NOT_INCLUDED`. For the second principal in
    the role binding, the key is `group:product-eng@example.com`, and the
    `membership` field in the value is set to `INCLUDED`.

    Messages:
      AdditionalProperty: An additional property for a MembershipsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MembershipsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MembershipsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudPolicytroubleshooterIamV3alphaAllowBindingExplanat
          ionAnnotatedAllowMembership attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaAllowBindingExplanationAnnotatedAllowMembership', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    allowAccessState = _messages.EnumField('AllowAccessStateValueValuesEnum', 1)
    combinedMembership = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaAllowBindingExplanationAnnotatedAllowMembership', 2)
    condition = _messages.MessageField('GoogleTypeExpr', 3)
    conditionExplanation = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaConditionExplanation', 4)
    memberships = _messages.MessageField('MembershipsValue', 5)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 6)
    role = _messages.StringField(7)
    rolePermission = _messages.EnumField('RolePermissionValueValuesEnum', 8)
    rolePermissionRelevance = _messages.EnumField('RolePermissionRelevanceValueValuesEnum', 9)