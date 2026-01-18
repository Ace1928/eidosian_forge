from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3alphaTroubleshootIamPolicyResponse(_messages.Message):
    """Response for TroubleshootIamPolicy.

  Enums:
    OverallAccessStateValueValuesEnum: Indicates whether the principal has the
      specified permission for the specified resource, based on evaluating all
      types of the applicable IAM policies.

  Fields:
    accessTuple: The access tuple from the request, including any provided
      context used to evaluate the condition.
    allowPolicyExplanation: An explanation of how the applicable IAM allow
      policies affect the final access state.
    denyPolicyExplanation: An explanation of how the applicable IAM deny
      policies affect the final access state.
    overallAccessState: Indicates whether the principal has the specified
      permission for the specified resource, based on evaluating all types of
      the applicable IAM policies.
    pabPolicyExplanation: An explanation of how the applicable Principal
      Access Boundary policies affect the final access state.
  """

    class OverallAccessStateValueValuesEnum(_messages.Enum):
        """Indicates whether the principal has the specified permission for the
    specified resource, based on evaluating all types of the applicable IAM
    policies.

    Values:
      OVERALL_ACCESS_STATE_UNSPECIFIED: Not specified.
      CAN_ACCESS: The principal has the permission.
      CANNOT_ACCESS: The principal doesn't have the permission.
      UNKNOWN_INFO: The principal might have the permission, but the sender
        can't access all of the information needed to fully evaluate the
        principal's access.
      UNKNOWN_CONDITIONAL: The principal might have the permission, but Policy
        Troubleshooter can't fully evaluate the principal's access because the
        sender didn't provide the required context to evaluate the condition.
    """
        OVERALL_ACCESS_STATE_UNSPECIFIED = 0
        CAN_ACCESS = 1
        CANNOT_ACCESS = 2
        UNKNOWN_INFO = 3
        UNKNOWN_CONDITIONAL = 4
    accessTuple = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaAccessTuple', 1)
    allowPolicyExplanation = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaAllowPolicyExplanation', 2)
    denyPolicyExplanation = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaDenyPolicyExplanation', 3)
    overallAccessState = _messages.EnumField('OverallAccessStateValueValuesEnum', 4)
    pabPolicyExplanation = _messages.MessageField('GoogleCloudPolicytroubleshooterIamV3alphaPABPolicyExplanation', 5)