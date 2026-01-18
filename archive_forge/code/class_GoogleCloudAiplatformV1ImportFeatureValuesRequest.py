from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ImportFeatureValuesRequest(_messages.Message):
    """Request message for FeaturestoreService.ImportFeatureValues.

  Fields:
    avroSource: A GoogleCloudAiplatformV1AvroSource attribute.
    bigquerySource: A GoogleCloudAiplatformV1BigQuerySource attribute.
    csvSource: A GoogleCloudAiplatformV1CsvSource attribute.
    disableIngestionAnalysis: If true, API doesn't start ingestion analysis
      pipeline.
    disableOnlineServing: If set, data will not be imported for online
      serving. This is typically used for backfilling, where Feature
      generation timestamps are not in the timestamp range needed for online
      serving.
    entityIdField: Source column that holds entity IDs. If not provided,
      entity IDs are extracted from the column named entity_id.
    featureSpecs: Required. Specifications defining which Feature values to
      import from the entity. The request fails if no feature_specs are
      provided, and having multiple feature_specs for one Feature is not
      allowed.
    featureTime: Single Feature timestamp for all entities being imported. The
      timestamp must not have higher than millisecond precision.
    featureTimeField: Source column that holds the Feature timestamp for all
      Feature values in each entity.
    workerCount: Specifies the number of workers that are used to write data
      to the Featurestore. Consider the online serving capacity that you
      require to achieve the desired import throughput without interfering
      with online serving. The value must be positive, and less than or equal
      to 100. If not set, defaults to using 1 worker. The low count ensures
      minimal impact on online serving performance.
  """
    avroSource = _messages.MessageField('GoogleCloudAiplatformV1AvroSource', 1)
    bigquerySource = _messages.MessageField('GoogleCloudAiplatformV1BigQuerySource', 2)
    csvSource = _messages.MessageField('GoogleCloudAiplatformV1CsvSource', 3)
    disableIngestionAnalysis = _messages.BooleanField(4)
    disableOnlineServing = _messages.BooleanField(5)
    entityIdField = _messages.StringField(6)
    featureSpecs = _messages.MessageField('GoogleCloudAiplatformV1ImportFeatureValuesRequestFeatureSpec', 7, repeated=True)
    featureTime = _messages.StringField(8)
    featureTimeField = _messages.StringField(9)
    workerCount = _messages.IntegerField(10, variant=_messages.Variant.INT32)