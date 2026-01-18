from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1betaExplainedPolicy(_messages.Message):
    """Details about how a specific IAM Policy contributed to the access check.

  Enums:
    AccessValueValuesEnum: Indicates whether _this policy_ provides the
      specified permission to the specified principal for the specified
      resource. This field does _not_ indicate whether the principal actually
      has the permission for the resource. There might be another policy that
      overrides this policy. To determine whether the principal actually has
      the permission, use the `access` field in the
      TroubleshootIamPolicyResponse.
    RelevanceValueValuesEnum: The relevance of this policy to the overall
      determination in the TroubleshootIamPolicyResponse. If the user who
      created the Replay does not have access to the policy, this field is
      omitted.

  Fields:
    access: Indicates whether _this policy_ provides the specified permission
      to the specified principal for the specified resource. This field does
      _not_ indicate whether the principal actually has the permission for the
      resource. There might be another policy that overrides this policy. To
      determine whether the principal actually has the permission, use the
      `access` field in the TroubleshootIamPolicyResponse.
    bindingExplanations: Details about how each binding in the policy affects
      the principal's ability, or inability, to use the permission for the
      resource. If the user who created the Replay does not have access to the
      policy, this field is omitted.
    fullResourceName: The full resource name that identifies the resource. For
      example, `//compute.googleapis.com/projects/my-project/zones/us-
      central1-a/instances/my-instance`. If the user who created the Replay
      does not have access to the policy, this field is omitted. For examples
      of full resource names for Google Cloud services, see
      https://cloud.google.com/iam/help/troubleshooter/full-resource-names.
    policy: The IAM policy attached to the resource. If the user who created
      the Replay does not have access to the policy, this field is empty.
    relevance: The relevance of this policy to the overall determination in
      the TroubleshootIamPolicyResponse. If the user who created the Replay
      does not have access to the policy, this field is omitted.
  """

    class AccessValueValuesEnum(_messages.Enum):
        """Indicates whether _this policy_ provides the specified permission to
    the specified principal for the specified resource. This field does _not_
    indicate whether the principal actually has the permission for the
    resource. There might be another policy that overrides this policy. To
    determine whether the principal actually has the permission, use the
    `access` field in the TroubleshootIamPolicyResponse.

    Values:
      ACCESS_STATE_UNSPECIFIED: Default value. This value is unused.
      GRANTED: The principal has the permission.
      NOT_GRANTED: The principal does not have the permission.
      UNKNOWN_CONDITIONAL: The principal has the permission only if a
        condition expression evaluates to `true`.
      UNKNOWN_INFO_DENIED: The user who created the Replay does not have
        access to all of the policies that Policy Simulator needs to evaluate.
    """
        ACCESS_STATE_UNSPECIFIED = 0
        GRANTED = 1
        NOT_GRANTED = 2
        UNKNOWN_CONDITIONAL = 3
        UNKNOWN_INFO_DENIED = 4

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of this policy to the overall determination in the
    TroubleshootIamPolicyResponse. If the user who created the Replay does not
    have access to the policy, this field is omitted.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Default value. This value is unused.
      NORMAL: The data point has a limited effect on the result. Changing the
        data point is unlikely to affect the overall determination.
      HIGH: The data point has a strong effect on the result. Changing the
        data point is likely to affect the overall determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        NORMAL = 1
        HIGH = 2
    access = _messages.EnumField('AccessValueValuesEnum', 1)
    bindingExplanations = _messages.MessageField('GoogleCloudPolicysimulatorV1betaBindingExplanation', 2, repeated=True)
    fullResourceName = _messages.StringField(3)
    policy = _messages.MessageField('GoogleIamV1Policy', 4)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 5)