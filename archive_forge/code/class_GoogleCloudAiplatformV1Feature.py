from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Feature(_messages.Message):
    """Feature Metadata information. For example, color is a feature that
  describes an apple.

  Enums:
    ValueTypeValueValuesEnum: Immutable. Only applicable for Vertex AI Feature
      Store (Legacy). Type of Feature value.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata to organize
      your Features. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      Feature (System labels are excluded)." System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable.

  Fields:
    createTime: Output only. Only applicable for Vertex AI Feature Store
      (Legacy). Timestamp when this EntityType was created.
    description: Description of the Feature.
    disableMonitoring: Optional. Only applicable for Vertex AI Feature Store
      (Legacy). If not set, use the monitoring_config defined for the
      EntityType this Feature belongs to. Only Features with type
      (Feature.ValueType) BOOL, STRING, DOUBLE or INT64 can enable monitoring.
      If set to true, all types of data monitoring are disabled despite the
      config on EntityType.
    etag: Used to perform a consistent read-modify-write updates. If not set,
      a blind "overwrite" update happens.
    labels: Optional. The labels with user-defined metadata to organize your
      Features. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      Feature (System labels are excluded)." System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable.
    monitoringStatsAnomalies: Output only. Only applicable for Vertex AI
      Feature Store (Legacy). The list of historical stats and anomalies with
      specified objectives.
    name: Immutable. Name of the Feature. Format: `projects/{project}/location
      s/{location}/featurestores/{featurestore}/entityTypes/{entity_type}/feat
      ures/{feature}` `projects/{project}/locations/{location}/featureGroups/{
      feature_group}/features/{feature}` The last part feature is assigned by
      the client. The feature can be up to 64 characters long and can consist
      only of ASCII Latin letters A-Z and a-z, underscore(_), and ASCII digits
      0-9 starting with a letter. The value will be unique given an entity
      type.
    pointOfContact: Entity responsible for maintaining this feature. Can be
      comma separated list of email addresses or URIs.
    updateTime: Output only. Only applicable for Vertex AI Feature Store
      (Legacy). Timestamp when this EntityType was most recently updated.
    valueType: Immutable. Only applicable for Vertex AI Feature Store
      (Legacy). Type of Feature value.
    versionColumnName: Only applicable for Vertex AI Feature Store. The name
      of the BigQuery Table/View column hosting data for this version. If no
      value is provided, will use feature_id.
  """

    class ValueTypeValueValuesEnum(_messages.Enum):
        """Immutable. Only applicable for Vertex AI Feature Store (Legacy). Type
    of Feature value.

    Values:
      VALUE_TYPE_UNSPECIFIED: The value type is unspecified.
      BOOL: Used for Feature that is a boolean.
      BOOL_ARRAY: Used for Feature that is a list of boolean.
      DOUBLE: Used for Feature that is double.
      DOUBLE_ARRAY: Used for Feature that is a list of double.
      INT64: Used for Feature that is INT64.
      INT64_ARRAY: Used for Feature that is a list of INT64.
      STRING: Used for Feature that is string.
      STRING_ARRAY: Used for Feature that is a list of String.
      BYTES: Used for Feature that is bytes.
    """
        VALUE_TYPE_UNSPECIFIED = 0
        BOOL = 1
        BOOL_ARRAY = 2
        DOUBLE = 3
        DOUBLE_ARRAY = 4
        INT64 = 5
        INT64_ARRAY = 6
        STRING = 7
        STRING_ARRAY = 8
        BYTES = 9

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata to organize your
    Features. Label keys and values can be no longer than 64 characters
    (Unicode codepoints), can only contain lowercase letters, numeric
    characters, underscores and dashes. International characters are allowed.
    See https://goo.gl/xmQnxf for more information on and examples of labels.
    No more than 64 user labels can be associated with one Feature (System
    labels are excluded)." System reserved label keys are prefixed with
    "aiplatform.googleapis.com/" and are immutable.

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
    description = _messages.StringField(2)
    disableMonitoring = _messages.BooleanField(3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    monitoringStatsAnomalies = _messages.MessageField('GoogleCloudAiplatformV1FeatureMonitoringStatsAnomaly', 6, repeated=True)
    name = _messages.StringField(7)
    pointOfContact = _messages.StringField(8)
    updateTime = _messages.StringField(9)
    valueType = _messages.EnumField('ValueTypeValueValuesEnum', 10)
    versionColumnName = _messages.StringField(11)