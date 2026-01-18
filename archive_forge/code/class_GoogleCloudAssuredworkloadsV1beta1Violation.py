from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1Violation(_messages.Message):
    """Workload monitoring Violation.

  Enums:
    StateValueValuesEnum: Output only. State of the violation
    ViolationTypeValueValuesEnum: Output only. Type of the violation

  Fields:
    acknowledged: A boolean that indicates if the violation is acknowledged
    acknowledgementTime: Optional. Timestamp when this violation was
      acknowledged first. Check exception_contexts to find the last time the
      violation was acknowledged when there are more than one violations. This
      field will be absent when acknowledged field is marked as false.
    associatedOrgPolicyViolationId: Optional. Output only. Violation Id of the
      org-policy violation due to which the resource violation is caused.
      Empty for org-policy violations.
    auditLogLink: Output only. Immutable. Audit Log Link for violated resource
      Format: https://console.cloud.google.com/logs/query;query={logName}{prot
      oPayload.resourceName}{timeRange}{folder}
    beginTime: Output only. Time of the event which triggered the Violation.
    category: Output only. Category under which this violation is mapped. e.g.
      Location, Service Usage, Access, Encryption, etc.
    description: Output only. Description for the Violation. e.g. OrgPolicy
      gcp.resourceLocations has non compliant value.
    exceptionAuditLogLink: Output only. Immutable. Audit Log link to find
      business justification provided for violation exception. Format: https:/
      /console.cloud.google.com/logs/query;query={logName}{protoPayload.resour
      ceName}{protoPayload.methodName}{timeRange}{organization}
    exceptionContexts: Output only. List of all the exception detail added for
      the violation.
    name: Output only. Immutable. Name of the Violation. Format: organizations
      /{organization}/locations/{location}/workloads/{workload_id}/violations/
      {violations_id}
    nonCompliantOrgPolicy: Output only. Immutable. Name of the OrgPolicy which
      was modified with non-compliant change and resulted this violation.
      Format: projects/{project_number}/policies/{constraint_name}
      folders/{folder_id}/policies/{constraint_name}
      organizations/{organization_id}/policies/{constraint_name}
    orgPolicyConstraint: Output only. Immutable. The org-policy-constraint
      that was incorrectly changed, which resulted in this violation.
    parentProjectNumber: Optional. Output only. Parent project number where
      resource is present. Empty for org-policy violations.
    remediation: Output only. Compliance violation remediation
    resolveTime: Output only. Time of the event which fixed the Violation. If
      the violation is ACTIVE this will be empty.
    resourceName: Optional. Output only. Name of the resource like
      //storage.googleapis.com/myprojectxyz-testbucket. Empty for org-policy
      violations.
    resourceType: Optional. Output only. Type of the resource like
      compute.googleapis.com/Disk, etc. Empty for org-policy violations.
    state: Output only. State of the violation
    updateTime: Output only. The last time when the Violation record was
      updated.
    violationType: Output only. Type of the violation
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the violation

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      RESOLVED: Violation is resolved.
      UNRESOLVED: Violation is Unresolved
      EXCEPTION: Violation is Exception
    """
        STATE_UNSPECIFIED = 0
        RESOLVED = 1
        UNRESOLVED = 2
        EXCEPTION = 3

    class ViolationTypeValueValuesEnum(_messages.Enum):
        """Output only. Type of the violation

    Values:
      VIOLATION_TYPE_UNSPECIFIED: Unspecified type.
      ORG_POLICY: Org Policy Violation.
      RESOURCE: Resource Violation.
    """
        VIOLATION_TYPE_UNSPECIFIED = 0
        ORG_POLICY = 1
        RESOURCE = 2
    acknowledged = _messages.BooleanField(1)
    acknowledgementTime = _messages.StringField(2)
    associatedOrgPolicyViolationId = _messages.StringField(3)
    auditLogLink = _messages.StringField(4)
    beginTime = _messages.StringField(5)
    category = _messages.StringField(6)
    description = _messages.StringField(7)
    exceptionAuditLogLink = _messages.StringField(8)
    exceptionContexts = _messages.MessageField('GoogleCloudAssuredworkloadsV1beta1ViolationExceptionContext', 9, repeated=True)
    name = _messages.StringField(10)
    nonCompliantOrgPolicy = _messages.StringField(11)
    orgPolicyConstraint = _messages.StringField(12)
    parentProjectNumber = _messages.StringField(13)
    remediation = _messages.MessageField('GoogleCloudAssuredworkloadsV1beta1ViolationRemediation', 14)
    resolveTime = _messages.StringField(15)
    resourceName = _messages.StringField(16)
    resourceType = _messages.StringField(17)
    state = _messages.EnumField('StateValueValuesEnum', 18)
    updateTime = _messages.StringField(19)
    violationType = _messages.EnumField('ViolationTypeValueValuesEnum', 20)