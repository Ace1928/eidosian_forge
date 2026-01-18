from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1p1beta1Finding(_messages.Message):
    """Security Command Center finding. A finding is a record of assessment
  data (security, risk, health or privacy) ingested into Security Command
  Center for presentation, notification, analysis, policy testing, and
  enforcement. For example, an XSS vulnerability in an App Engine application
  is a finding.

  Enums:
    SeverityValueValuesEnum: The severity of the finding. This field is
      managed by the source that writes the finding.
    StateValueValuesEnum: The state of the finding.

  Messages:
    SourcePropertiesValue: Source specific properties. These properties are
      managed by the source that writes the finding. The key names in the
      source_properties map must be between 1 and 255 characters, and must
      start with a letter and contain alphanumeric characters or underscores
      only.

  Fields:
    canonicalName: The canonical name of the finding. It's either "organizatio
      ns/{organization_id}/sources/{source_id}/findings/{finding_id}",
      "folders/{folder_id}/sources/{source_id}/findings/{finding_id}" or
      "projects/{project_number}/sources/{source_id}/findings/{finding_id}",
      depending on the closest CRM ancestor of the resource associated with
      the finding.
    category: The additional taxonomy group within findings from a given
      source. This field is immutable after creation time. Example:
      "XSS_FLASH_INJECTION"
    createTime: The time at which the finding was created in Security Command
      Center.
    eventTime: The time at which the event took place, or when an update to
      the finding occurred. For example, if the finding represents an open
      firewall it would capture the time the detector believes the firewall
      became open. The accuracy is determined by the detector. If the finding
      were to be resolved afterward, this time would reflect when the finding
      was resolved. Must not be set to a value greater than the current
      timestamp.
    externalUri: The URI that, if available, points to a web page outside of
      Security Command Center where additional information about the finding
      can be found. This field is guaranteed to be either empty or a well
      formed URL.
    name: The relative resource name of this finding. See:
      https://cloud.google.com/apis/design/resource_names#relative_resource_na
      me Example: "organizations/{organization_id}/sources/{source_id}/finding
      s/{finding_id}"
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
    securityMarks: Output only. User specified security marks. These marks are
      entirely managed by the user and come from the SecurityMarks resource
      that belongs to the finding.
    severity: The severity of the finding. This field is managed by the source
      that writes the finding.
    sourceProperties: Source specific properties. These properties are managed
      by the source that writes the finding. The key names in the
      source_properties map must be between 1 and 255 characters, and must
      start with a letter and contain alphanumeric characters or underscores
      only.
    state: The state of the finding.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """The severity of the finding. This field is managed by the source that
    writes the finding.

    Values:
      SEVERITY_UNSPECIFIED: No severity specified. The default value.
      CRITICAL: Critical severity.
      HIGH: High severity.
      MEDIUM: Medium severity.
      LOW: Low severity.
    """
        SEVERITY_UNSPECIFIED = 0
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4

    class StateValueValuesEnum(_messages.Enum):
        """The state of the finding.

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
    canonicalName = _messages.StringField(1)
    category = _messages.StringField(2)
    createTime = _messages.StringField(3)
    eventTime = _messages.StringField(4)
    externalUri = _messages.StringField(5)
    name = _messages.StringField(6)
    parent = _messages.StringField(7)
    resourceName = _messages.StringField(8)
    securityMarks = _messages.MessageField('GoogleCloudSecuritycenterV1p1beta1SecurityMarks', 9)
    severity = _messages.EnumField('SeverityValueValuesEnum', 10)
    sourceProperties = _messages.MessageField('SourcePropertiesValue', 11)
    state = _messages.EnumField('StateValueValuesEnum', 12)