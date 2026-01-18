from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScan(_messages.Message):
    """Represents a user-visible job which provides the insights for the
  related data source.For example: Data Quality: generates queries based on
  the rules and runs against the data to get data quality check results. Data
  Profile: analyzes the data in table(s) and generates insights about the
  structure, content and relationships (such as null percent, cardinality,
  min/max/mean, etc).

  Enums:
    StateValueValuesEnum: Output only. Current state of the DataScan.
    TypeValueValuesEnum: Output only. The type of DataScan.

  Messages:
    LabelsValue: Optional. User-defined labels for the scan.

  Fields:
    createTime: Output only. The time when the scan was created.
    data: Required. The data source for DataScan.
    dataProfileResult: Output only. The result of the data profile scan.
    dataProfileSpec: DataProfileScan related setting.
    dataQualityResult: Output only. The result of the data quality scan.
    dataQualitySpec: DataQualityScan related setting.
    description: Optional. Description of the scan. Must be between 1-1024
      characters.
    displayName: Optional. User friendly display name. Must be between 1-256
      characters.
    executionSpec: Optional. DataScan execution settings.If not specified, the
      fields in it will use their default values.
    executionStatus: Output only. Status of the data scan execution.
    labels: Optional. User-defined labels for the scan.
    name: Output only. The relative resource name of the scan, of the form:
      projects/{project}/locations/{location_id}/dataScans/{datascan_id},
      where project refers to a project_id or project_number and location_id
      refers to a GCP region.
    state: Output only. Current state of the DataScan.
    type: Output only. The type of DataScan.
    uid: Output only. System generated globally unique ID for the scan. This
      ID will be different if the scan is deleted and re-created with the same
      name.
    updateTime: Output only. The time when the scan was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the DataScan.

    Values:
      STATE_UNSPECIFIED: State is not specified.
      ACTIVE: Resource is active, i.e., ready to use.
      CREATING: Resource is under creation.
      DELETING: Resource is under deletion.
      ACTION_REQUIRED: Resource is active but has unresolved actions.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        DELETING = 3
        ACTION_REQUIRED = 4

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. The type of DataScan.

    Values:
      DATA_SCAN_TYPE_UNSPECIFIED: The DataScan type is unspecified.
      DATA_QUALITY: Data Quality scan.
      DATA_PROFILE: Data Profile scan.
    """
        DATA_SCAN_TYPE_UNSPECIFIED = 0
        DATA_QUALITY = 1
        DATA_PROFILE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-defined labels for the scan.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    data = _messages.MessageField('GoogleCloudDataplexV1DataSource', 2)
    dataProfileResult = _messages.MessageField('GoogleCloudDataplexV1DataProfileResult', 3)
    dataProfileSpec = _messages.MessageField('GoogleCloudDataplexV1DataProfileSpec', 4)
    dataQualityResult = _messages.MessageField('GoogleCloudDataplexV1DataQualityResult', 5)
    dataQualitySpec = _messages.MessageField('GoogleCloudDataplexV1DataQualitySpec', 6)
    description = _messages.StringField(7)
    displayName = _messages.StringField(8)
    executionSpec = _messages.MessageField('GoogleCloudDataplexV1DataScanExecutionSpec', 9)
    executionStatus = _messages.MessageField('GoogleCloudDataplexV1DataScanExecutionStatus', 10)
    labels = _messages.MessageField('LabelsValue', 11)
    name = _messages.StringField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    type = _messages.EnumField('TypeValueValuesEnum', 14)
    uid = _messages.StringField(15)
    updateTime = _messages.StringField(16)