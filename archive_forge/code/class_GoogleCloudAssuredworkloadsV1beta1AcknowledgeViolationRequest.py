from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1AcknowledgeViolationRequest(_messages.Message):
    """Request for acknowledging the violation

  Enums:
    AcknowledgeTypeValueValuesEnum: Optional. Acknowledge type of specified
      violation.

  Fields:
    acknowledgeType: Optional. Acknowledge type of specified violation.
    comment: Required. Business justification explaining the need for
      violation acknowledgement
    nonCompliantOrgPolicy: Optional. This field is deprecated and will be
      removed in future version of the API. Name of the OrgPolicy which was
      modified with non-compliant change and resulted in this violation.
      Format: projects/{project_number}/policies/{constraint_name}
      folders/{folder_id}/policies/{constraint_name}
      organizations/{organization_id}/policies/{constraint_name}
  """

    class AcknowledgeTypeValueValuesEnum(_messages.Enum):
        """Optional. Acknowledge type of specified violation.

    Values:
      ACKNOWLEDGE_TYPE_UNSPECIFIED: Acknowledge type unspecified.
      SINGLE_VIOLATION: Acknowledge only the specific violation.
      EXISTING_CHILD_RESOURCE_VIOLATIONS: Acknowledge specified orgPolicy
        violation and also associated resource violations.
    """
        ACKNOWLEDGE_TYPE_UNSPECIFIED = 0
        SINGLE_VIOLATION = 1
        EXISTING_CHILD_RESOURCE_VIOLATIONS = 2
    acknowledgeType = _messages.EnumField('AcknowledgeTypeValueValuesEnum', 1)
    comment = _messages.StringField(2)
    nonCompliantOrgPolicy = _messages.StringField(3)