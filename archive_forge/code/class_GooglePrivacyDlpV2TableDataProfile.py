from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TableDataProfile(_messages.Message):
    """The profile for a scanned table.

  Enums:
    EncryptionStatusValueValuesEnum: How the table is encrypted.
    ResourceVisibilityValueValuesEnum: How broadly a resource has been shared.
    StateValueValuesEnum: State of a profile.

  Messages:
    ResourceLabelsValue: The labels applied to the resource at the time the
      profile was generated.

  Fields:
    configSnapshot: The snapshot of the configurations used to generate the
      profile.
    createTime: The time at which the table was created.
    dataRiskLevel: The data risk level of this table.
    dataSourceType: The resource type that was profiled.
    datasetId: If the resource is BigQuery, the dataset ID.
    datasetLocation: If supported, the location where the dataset's data is
      stored. See https://cloud.google.com/bigquery/docs/locations for
      supported locations.
    datasetProjectId: The Google Cloud project ID that owns the resource.
    encryptionStatus: How the table is encrypted.
    expirationTime: Optional. The time when this table expires.
    failedColumnCount: The number of columns skipped in the table because of
      an error.
    fullResource: The resource name of the resource profiled.
      https://cloud.google.com/apis/design/resource_names#full_resource_name
    lastModifiedTime: The time when this table was last modified
    name: The name of the profile.
    otherInfoTypes: Other infoTypes found in this table's data.
    predictedInfoTypes: The infoTypes predicted from this table's data.
    profileLastGenerated: The last time the profile was generated.
    profileStatus: Success or error status from the most recent profile
      generation attempt. May be empty if the profile is still being
      generated.
    projectDataProfile: The resource name to the project data profile for this
      table.
    resourceLabels: The labels applied to the resource at the time the profile
      was generated.
    resourceVisibility: How broadly a resource has been shared.
    rowCount: Number of rows in the table when the profile was generated. This
      will not be populated for BigLake tables.
    scannedColumnCount: The number of columns profiled in the table.
    sensitivityScore: The sensitivity score of this table.
    state: State of a profile.
    tableId: If the resource is BigQuery, the BigQuery table ID.
    tableSizeBytes: The size of the table when the profile was generated.
  """

    class EncryptionStatusValueValuesEnum(_messages.Enum):
        """How the table is encrypted.

    Values:
      ENCRYPTION_STATUS_UNSPECIFIED: Unused.
      ENCRYPTION_GOOGLE_MANAGED: Google manages server-side encryption keys on
        your behalf.
      ENCRYPTION_CUSTOMER_MANAGED: Customer provides the key.
    """
        ENCRYPTION_STATUS_UNSPECIFIED = 0
        ENCRYPTION_GOOGLE_MANAGED = 1
        ENCRYPTION_CUSTOMER_MANAGED = 2

    class ResourceVisibilityValueValuesEnum(_messages.Enum):
        """How broadly a resource has been shared.

    Values:
      RESOURCE_VISIBILITY_UNSPECIFIED: Unused.
      RESOURCE_VISIBILITY_PUBLIC: Visible to any user.
      RESOURCE_VISIBILITY_RESTRICTED: Visible only to specific users.
    """
        RESOURCE_VISIBILITY_UNSPECIFIED = 0
        RESOURCE_VISIBILITY_PUBLIC = 1
        RESOURCE_VISIBILITY_RESTRICTED = 2

    class StateValueValuesEnum(_messages.Enum):
        """State of a profile.

    Values:
      STATE_UNSPECIFIED: Unused.
      RUNNING: The profile is currently running. Once a profile has finished
        it will transition to DONE.
      DONE: The profile is no longer generating. If profile_status.status.code
        is 0, the profile succeeded, otherwise, it failed.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        DONE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceLabelsValue(_messages.Message):
        """The labels applied to the resource at the time the profile was
    generated.

    Messages:
      AdditionalProperty: An additional property for a ResourceLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ResourceLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    configSnapshot = _messages.MessageField('GooglePrivacyDlpV2DataProfileConfigSnapshot', 1)
    createTime = _messages.StringField(2)
    dataRiskLevel = _messages.MessageField('GooglePrivacyDlpV2DataRiskLevel', 3)
    dataSourceType = _messages.MessageField('GooglePrivacyDlpV2DataSourceType', 4)
    datasetId = _messages.StringField(5)
    datasetLocation = _messages.StringField(6)
    datasetProjectId = _messages.StringField(7)
    encryptionStatus = _messages.EnumField('EncryptionStatusValueValuesEnum', 8)
    expirationTime = _messages.StringField(9)
    failedColumnCount = _messages.IntegerField(10)
    fullResource = _messages.StringField(11)
    lastModifiedTime = _messages.StringField(12)
    name = _messages.StringField(13)
    otherInfoTypes = _messages.MessageField('GooglePrivacyDlpV2OtherInfoTypeSummary', 14, repeated=True)
    predictedInfoTypes = _messages.MessageField('GooglePrivacyDlpV2InfoTypeSummary', 15, repeated=True)
    profileLastGenerated = _messages.StringField(16)
    profileStatus = _messages.MessageField('GooglePrivacyDlpV2ProfileStatus', 17)
    projectDataProfile = _messages.StringField(18)
    resourceLabels = _messages.MessageField('ResourceLabelsValue', 19)
    resourceVisibility = _messages.EnumField('ResourceVisibilityValueValuesEnum', 20)
    rowCount = _messages.IntegerField(21)
    scannedColumnCount = _messages.IntegerField(22)
    sensitivityScore = _messages.MessageField('GooglePrivacyDlpV2SensitivityScore', 23)
    state = _messages.EnumField('StateValueValuesEnum', 24)
    tableId = _messages.StringField(25)
    tableSizeBytes = _messages.IntegerField(26)