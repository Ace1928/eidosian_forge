from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceOSPoliciesComplianceOSPolicyCompliance(_messages.Message):
    """Compliance data for an OS policy

  Enums:
    StateValueValuesEnum: Compliance state of the OS policy.

  Fields:
    osPolicyAssignment: Reference to the `OSPolicyAssignment` API resource
      that the `OSPolicy` belongs to. Format: `projects/{project_number}/locat
      ions/{location}/osPolicyAssignments/{os_policy_assignment_id@revision_id
      }`
    osPolicyId: The OS policy id
    osPolicyResourceCompliances: Compliance data for each `OSPolicyResource`
      that is applied to the VM.
    state: Compliance state of the OS policy.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Compliance state of the OS policy.

    Values:
      OS_POLICY_COMPLIANCE_STATE_UNSPECIFIED: Default value. This value is
        unused.
      COMPLIANT: Compliant state.
      NON_COMPLIANT: Non-compliant state
      UNKNOWN: Unknown compliance state.
      NO_OS_POLICIES_APPLICABLE: No applicable OS policies were found for the
        instance. This state is only applicable to the instance.
    """
        OS_POLICY_COMPLIANCE_STATE_UNSPECIFIED = 0
        COMPLIANT = 1
        NON_COMPLIANT = 2
        UNKNOWN = 3
        NO_OS_POLICIES_APPLICABLE = 4
    osPolicyAssignment = _messages.StringField(1)
    osPolicyId = _messages.StringField(2)
    osPolicyResourceCompliances = _messages.MessageField('OSPolicyResourceCompliance', 3, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 4)