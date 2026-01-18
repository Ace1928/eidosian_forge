from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyAssignmentReportOSPolicyComplianceOSPolicyResourceCompliance(_messages.Message):
    """Compliance data for an OS policy resource.

  Enums:
    ComplianceStateValueValuesEnum: The compliance state of the resource.

  Fields:
    complianceState: The compliance state of the resource.
    complianceStateReason: A reason for the resource to be in the given
      compliance state. This field is always populated when `compliance_state`
      is `UNKNOWN`. The following values are supported when `compliance_state
      == UNKNOWN` * `execution-errors`: Errors were encountered by the agent
      while executing the resource and the compliance state couldn't be
      determined. * `execution-skipped-by-agent`: Resource execution was
      skipped by the agent because errors were encountered while executing
      prior resources in the OS policy. * `os-policy-execution-attempt-
      failed`: The execution of the OS policy containing this resource failed
      and the compliance state couldn't be determined.
    configSteps: Ordered list of configuration completed by the agent for the
      OS policy resource.
    execResourceOutput: ExecResource specific output.
    osPolicyResourceId: The ID of the OS policy resource.
  """

    class ComplianceStateValueValuesEnum(_messages.Enum):
        """The compliance state of the resource.

    Values:
      UNKNOWN: The resource is in an unknown compliance state. To get more
        details about why the policy is in this state, review the output of
        the `compliance_state_reason` field.
      COMPLIANT: Resource is compliant.
      NON_COMPLIANT: Resource is non-compliant.
    """
        UNKNOWN = 0
        COMPLIANT = 1
        NON_COMPLIANT = 2
    complianceState = _messages.EnumField('ComplianceStateValueValuesEnum', 1)
    complianceStateReason = _messages.StringField(2)
    configSteps = _messages.MessageField('OSPolicyAssignmentReportOSPolicyComplianceOSPolicyResourceComplianceOSPolicyResourceConfigStep', 3, repeated=True)
    execResourceOutput = _messages.MessageField('OSPolicyAssignmentReportOSPolicyComplianceOSPolicyResourceComplianceExecResourceOutput', 4)
    osPolicyResourceId = _messages.StringField(5)