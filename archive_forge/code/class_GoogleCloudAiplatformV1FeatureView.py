from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureView(_messages.Message):
    """FeatureView is representation of values that the FeatureOnlineStore will
  serve based on its syncConfig.

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
    syncConfig: Configures when data is to be synced/updated for this
      FeatureView. At the end of the sync the latest featureValues for each
      entityId of this FeatureView are made ready for online serving.
    updateTime: Output only. Timestamp when this FeatureView was last updated.
  """

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
    bigQuerySource = _messages.MessageField('GoogleCloudAiplatformV1FeatureViewBigQuerySource', 1)
    createTime = _messages.StringField(2)
    etag = _messages.StringField(3)
    featureRegistrySource = _messages.MessageField('GoogleCloudAiplatformV1FeatureViewFeatureRegistrySource', 4)
    indexConfig = _messages.MessageField('GoogleCloudAiplatformV1FeatureViewIndexConfig', 5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    syncConfig = _messages.MessageField('GoogleCloudAiplatformV1FeatureViewSyncConfig', 8)
    updateTime = _messages.StringField(9)