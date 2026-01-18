from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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