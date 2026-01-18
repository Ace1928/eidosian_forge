from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureView(_messages.Message):
    """FeatureView is representation of values that the FeatureOnlineStore will
  serve based on its syncConfig.

  Enums:
    ServiceAgentTypeValueValuesEnum: Optional. Service agent type used during
      data sync. By default, the Vertex AI Service Agent is used. When using
      an IAM Policy to isolate this FeatureView within a project, a separate
      service account should be provisioned by setting this field to
      `SERVICE_AGENT_TYPE_FEATURE_VIEW`. This will generate a separate service
      account to access the BigQuery source table.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata to organize
      your FeatureViews. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      FeatureOnlineStore(System labels are excluded)." System reserved label
      keys are prefixed with "aiplatform.googleapis.com/" and are immutable.

  Fields:
    bigQuerySource: Optional. Configures how data is supposed to be extracted
      from a BigQuery source to be loaded onto the FeatureOnlineStore.
    createTime: Output only. Timestamp when this FeatureView was created.
    etag: Optional. Used to perform consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    featureRegistrySource: Optional. Configures the features from a Feature
      Registry source that need to be loaded onto the FeatureOnlineStore.
    indexConfig: Optional. Configuration for index preparation for vector
      search. It contains the required configurations to create an index from
      source data, so that approximate nearest neighbor (a.k.a ANN) algorithms
      search can be performed during online serving.
    labels: Optional. The labels with user-defined metadata to organize your
      FeatureViews. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      FeatureOnlineStore(System labels are excluded)." System reserved label
      keys are prefixed with "aiplatform.googleapis.com/" and are immutable.
    name: Identifier. Name of the FeatureView. Format: `projects/{project}/loc
      ations/{location}/featureOnlineStores/{feature_online_store}/featureView
      s/{feature_view}`
    serviceAccountEmail: Output only. A Service Account unique to this
      FeatureView. The role bigquery.dataViewer should be granted to this
      service account to allow Vertex AI Feature Store to sync data to the
      online store.
    serviceAgentType: Optional. Service agent type used during data sync. By
      default, the Vertex AI Service Agent is used. When using an IAM Policy
      to isolate this FeatureView within a project, a separate service account
      should be provisioned by setting this field to
      `SERVICE_AGENT_TYPE_FEATURE_VIEW`. This will generate a separate service
      account to access the BigQuery source table.
    syncConfig: Configures when data is to be synced/updated for this
      FeatureView. At the end of the sync the latest featureValues for each
      entityId of this FeatureView are made ready for online serving.
    updateTime: Output only. Timestamp when this FeatureView was last updated.
    vectorSearchConfig: Optional. Deprecated: please use
      FeatureView.index_config instead.
  """

    class ServiceAgentTypeValueValuesEnum(_messages.Enum):
        """Optional. Service agent type used during data sync. By default, the
    Vertex AI Service Agent is used. When using an IAM Policy to isolate this
    FeatureView within a project, a separate service account should be
    provisioned by setting this field to `SERVICE_AGENT_TYPE_FEATURE_VIEW`.
    This will generate a separate service account to access the BigQuery
    source table.

    Values:
      SERVICE_AGENT_TYPE_UNSPECIFIED: By default, the project-level Vertex AI
        Service Agent is enabled.
      SERVICE_AGENT_TYPE_PROJECT: Indicates the project-level Vertex AI
        Service Agent (https://cloud.google.com/vertex-ai/docs/general/access-
        control#service-agents) will be used during sync jobs.
      SERVICE_AGENT_TYPE_FEATURE_VIEW: Enable a FeatureView service account to
        be created by Vertex AI and output in the field
        `service_account_email`. This service account will be used to read
        from the source BigQuery table during sync.
    """
        SERVICE_AGENT_TYPE_UNSPECIFIED = 0
        SERVICE_AGENT_TYPE_PROJECT = 1
        SERVICE_AGENT_TYPE_FEATURE_VIEW = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata to organize your
    FeatureViews. Label keys and values can be no longer than 64 characters
    (Unicode codepoints), can only contain lowercase letters, numeric
    characters, underscores and dashes. International characters are allowed.
    See https://goo.gl/xmQnxf for more information on and examples of labels.
    No more than 64 user labels can be associated with one
    FeatureOnlineStore(System labels are excluded)." System reserved label
    keys are prefixed with "aiplatform.googleapis.com/" and are immutable.

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
    bigQuerySource = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewBigQuerySource', 1)
    createTime = _messages.StringField(2)
    etag = _messages.StringField(3)
    featureRegistrySource = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewFeatureRegistrySource', 4)
    indexConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewIndexConfig', 5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    serviceAccountEmail = _messages.StringField(8)
    serviceAgentType = _messages.EnumField('ServiceAgentTypeValueValuesEnum', 9)
    syncConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewSyncConfig', 10)
    updateTime = _messages.StringField(11)
    vectorSearchConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewVectorSearchConfig', 12)