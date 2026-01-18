from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudPolicytroubleshooterV1betaTroubleshootIamPolicyResponse(_messages.Message):
    """Response for TroubleshootIamPolicy.

  Enums:
    AccessValueValuesEnum: Indicates whether the member has the specified
      permission for the specified resource, based on evaluating all of the
      applicable policies.

  Fields:
    access: Indicates whether the member has the specified permission for the
      specified resource, based on evaluating all of the applicable policies.
    explainedPolicies: List of IAM policies that were evaluated to check the
      member's permissions, with annotations to indicate how each policy
      contributed to the final result. The list of policies can include the
      policy for the resource itself. It can also include policies that are
      inherited from higher levels of the resource hierarchy, including the
      organization, the folder, and the project. To learn more about the
      resource hierarchy, see https://cloud.google.com/iam/help/resource-
      hierarchy.
  """

    class AccessValueValuesEnum(_messages.Enum):
        """Indicates whether the member has the specified permission for the
    specified resource, based on evaluating all of the applicable policies.

    Values:
      ACCESS_STATE_UNSPECIFIED: Reserved for future use.
      GRANTED: The member has the permission.
      NOT_GRANTED: The member does not have the permission.
      UNKNOWN_CONDITIONAL: The member has the permission only if a condition
        expression evaluates to `true`.
      UNKNOWN_INFO_DENIED: The sender of the request does not have access to
        all of the policies that Policy Troubleshooter needs to evaluate.
    """
        ACCESS_STATE_UNSPECIFIED = 0
        GRANTED = 1
        NOT_GRANTED = 2
        UNKNOWN_CONDITIONAL = 3
        UNKNOWN_INFO_DENIED = 4
    access = _messages.EnumField('AccessValueValuesEnum', 1)
    explainedPolicies = _messages.MessageField('GoogleCloudPolicytroubleshooterV1betaExplainedPolicy', 2, repeated=True)