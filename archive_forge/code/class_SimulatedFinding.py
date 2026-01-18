from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SimulatedFinding(_messages.Message):
    """A subset of the fields of the Security Center Finding proto. The minimum
  set of fields needed to represent a simulated finding from a SHA custom
  module.

  Enums:
    FindingClassValueValuesEnum: The class of the finding.
    SeverityValueValuesEnum: The severity of the finding. This field is
      managed by the source that writes the finding.
    StateValueValuesEnum: Output only. The state of the finding.

  Messages:
    SourcePropertiesValue: Source specific properties. These properties are
      managed by the source that writes the finding. The key names in the
      source_properties map must be between 1 and 255 characters, and must
      start with a letter and contain alphanumeric characters or underscores
      only.

  Fields:
    category: The additional taxonomy group within findings from a given
      source. This field is immutable after creation time. Example:
      "XSS_FLASH_INJECTION"
    eventTime: The time the finding was first detected. If an existing finding
      is updated, then this is the time the update occurred. For example, if
      the finding represents an open firewall, this property captures the time
      the detector believes the firewall became open. The accuracy is
      determined by the detector. If the finding is later resolved, then this
      time reflects when the finding was resolved. This must not be set to a
      value greater than the current timestamp.
    findingClass: The class of the finding.
    name: Identifier. The [relative resource name](https://cloud.google.com/ap
      is/design/resource_names#relative_resource_name) of the finding.
      Example: "organizations/{organization_id}/sources/{source_id}/findings/{
      finding_id}",
      "folders/{folder_id}/sources/{source_id}/findings/{finding_id}",
      "projects/{project_id}/sources/{source_id}/findings/{finding_id}".
    parent: The relative resource name of the source the finding belongs to.
      See: https://cloud.google.com/apis/design/resource_names#relative_resour
      ce_name This field is immutable after creation time. For example:
      "organizations/{organization_id}/sources/{source_id}"
    resourceName: For findings on Google Cloud resources, the full resource
      name of the Google Cloud resource this finding is for. See:
      https://cloud.google.com/apis/design/resource_names#full_resource_name
      When the finding is for a non-Google Cloud resource, the resourceName
      can be a customer or partner defined string. This field is immutable
      after creation time.
    severity: The severity of the finding. This field is managed by the source
      that writes the finding.
    sourceProperties: Source specific properties. These properties are managed
      by the source that writes the finding. The key names in the
      source_properties map must be between 1 and 255 characters, and must
      start with a letter and contain alphanumeric characters or underscores
      only.
    state: Output only. The state of the finding.
  """

    class FindingClassValueValuesEnum(_messages.Enum):
        """The class of the finding.

    Values:
      FINDING_CLASS_UNSPECIFIED: Unspecified finding class.
      THREAT: Describes unwanted or malicious activity.
      VULNERABILITY: Describes a potential weakness in software that increases
        risk to Confidentiality & Integrity & Availability.
      MISCONFIGURATION: Describes a potential weakness in cloud resource/asset
        configuration that increases risk.
      OBSERVATION: Describes a security observation that is for informational
        purposes.
      SCC_ERROR: Describes an error that prevents some SCC functionality.
      POSTURE_VIOLATION: Describes a potential security risk due to a change
        in the security posture.
    """
        FINDING_CLASS_UNSPECIFIED = 0
        THREAT = 1
        VULNERABILITY = 2
        MISCONFIGURATION = 3
        OBSERVATION = 4
        SCC_ERROR = 5
        POSTURE_VIOLATION = 6

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity of the finding. This field is managed by the source that
    writes the finding.

    Values:
      SEVERITY_UNSPECIFIED: This value is used for findings when a source
        doesn't write a severity value.
      CRITICAL: Vulnerability: A critical vulnerability is easily discoverable
        by an external actor, exploitable, and results in the direct ability
        to execute arbitrary code, exfiltrate data, and otherwise gain
        additional access and privileges to cloud resources and workloads.
        Examples include publicly accessible unprotected user data and public
        SSH access with weak or no passwords. Threat: Indicates a threat that
        is able to access, modify, or delete data or execute unauthorized code
        within existing resources.
      HIGH: Vulnerability: A high risk vulnerability can be easily discovered
        and exploited in combination with other vulnerabilities in order to
        gain direct access and the ability to execute arbitrary code,
        exfiltrate data, and otherwise gain additional access and privileges
        to cloud resources and workloads. An example is a database with weak
        or no passwords that is only accessible internally. This database
        could easily be compromised by an actor that had access to the
        internal network. Threat: Indicates a threat that is able to create
        new computational resources in an environment but not able to access
        data or execute code in existing resources.
      MEDIUM: Vulnerability: A medium risk vulnerability could be used by an
        actor to gain access to resources or privileges that enable them to
        eventually (through multiple steps or a complex exploit) gain access
        and the ability to execute arbitrary code or exfiltrate data. An
        example is a service account with access to more projects than it
        should have. If an actor gains access to the service account, they
        could potentially use that access to manipulate a project the service
        account was not intended to. Threat: Indicates a threat that is able
        to cause operational impact but may not access data or execute
        unauthorized code.
      LOW: Vulnerability: A low risk vulnerability hampers a security
        organization's ability to detect vulnerabilities or active threats in
        their deployment, or prevents the root cause investigation of security
        issues. An example is monitoring and logs being disabled for resource
        configurations and access. Threat: Indicates a threat that has
        obtained minimal access to an environment but is not able to access
        data, execute code, or create resources.
    """
        SEVERITY_UNSPECIFIED = 0
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the finding.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      ACTIVE: The finding requires attention and has not been addressed yet.
      INACTIVE: The finding has been fixed, triaged as a non-issue or
        otherwise addressed and is no longer active.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SourcePropertiesValue(_messages.Message):
        """Source specific properties. These properties are managed by the source
    that writes the finding. The key names in the source_properties map must
    be between 1 and 255 characters, and must start with a letter and contain
    alphanumeric characters or underscores only.

    Messages:
      AdditionalProperty: An additional property for a SourcePropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        SourcePropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SourcePropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    category = _messages.StringField(1)
    eventTime = _messages.StringField(2)
    findingClass = _messages.EnumField('FindingClassValueValuesEnum', 3)
    name = _messages.StringField(4)
    parent = _messages.StringField(5)
    resourceName = _messages.StringField(6)
    severity = _messages.EnumField('SeverityValueValuesEnum', 7)
    sourceProperties = _messages.MessageField('SourcePropertiesValue', 8)
    state = _messages.EnumField('StateValueValuesEnum', 9)