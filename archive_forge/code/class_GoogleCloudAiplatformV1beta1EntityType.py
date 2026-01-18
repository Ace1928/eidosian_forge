from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1EntityType(_messages.Message):
    """An entity type is a type of object in a system that needs to be modeled
  and have stored information about. For example, driver is an entity type,
  and driver0 is an instance of an entity type driver.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata to organize
      your EntityTypes. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      EntityType (System labels are excluded)." System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable.

  Fields:
    createTime: Output only. Timestamp when this EntityType was created.
    description: Optional. Description of the EntityType.
    etag: Optional. Used to perform a consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    labels: Optional. The labels with user-defined metadata to organize your
      EntityTypes. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      EntityType (System labels are excluded)." System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable.
    monitoringConfig: Optional. The default monitoring configuration for all
      Features with value type (Feature.ValueType) BOOL, STRING, DOUBLE or
      INT64 under this EntityType. If this is populated with
      [FeaturestoreMonitoringConfig.monitoring_interval] specified, snapshot
      analysis monitoring is enabled. Otherwise, snapshot analysis monitoring
      is disabled.
    name: Immutable. Name of the EntityType. Format: `projects/{project}/locat
      ions/{location}/featurestores/{featurestore}/entityTypes/{entity_type}`
      The last part entity_type is assigned by the client. The entity_type can
      be up to 64 characters long and can consist only of ASCII Latin letters
      A-Z and a-z and underscore(_), and ASCII digits 0-9 starting with a
      letter. The value will be unique given a featurestore.
    offlineStorageTtlDays: Optional. Config for data retention policy in
      offline storage. TTL in days for feature values that will be stored in
      offline storage. The Feature Store offline storage periodically removes
      obsolete feature values older than `offline_storage_ttl_days` since the
      feature generation time. If unset (or explicitly set to 0), default to
      4000 days TTL.
    updateTime: Output only. Timestamp when this EntityType was most recently
      updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata to organize your
    EntityTypes. Label keys and values can be no longer than 64 characters
    (Unicode codepoints), can only contain lowercase letters, numeric
    characters, underscores and dashes. International characters are allowed.
    See https://goo.gl/xmQnxf for more information on and examples of labels.
    No more than 64 user labels can be associated with one EntityType (System
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
    etag = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    monitoringConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1FeaturestoreMonitoringConfig', 5)
    name = _messages.StringField(6)
    offlineStorageTtlDays = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    updateTime = _messages.StringField(8)