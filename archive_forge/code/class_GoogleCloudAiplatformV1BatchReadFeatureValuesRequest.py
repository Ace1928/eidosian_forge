from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BatchReadFeatureValuesRequest(_messages.Message):
    """Request message for FeaturestoreService.BatchReadFeatureValues.

  Fields:
    bigqueryReadInstances: Similar to csv_read_instances, but from BigQuery
      source.
    csvReadInstances: Each read instance consists of exactly one read
      timestamp and one or more entity IDs identifying entities of the
      corresponding EntityTypes whose Features are requested. Each output
      instance contains Feature values of requested entities concatenated
      together as of the read time. An example read instance may be
      `foo_entity_id, bar_entity_id, 2020-01-01T10:00:00.123Z`. An example
      output instance may be `foo_entity_id, bar_entity_id,
      2020-01-01T10:00:00.123Z, foo_entity_feature1_value,
      bar_entity_feature2_value`. Timestamp in each read instance must be
      millisecond-aligned. `csv_read_instances` are read instances stored in a
      plain-text CSV file. The header should be: [ENTITY_TYPE_ID1],
      [ENTITY_TYPE_ID2], ..., timestamp The columns can be in any order.
      Values in the timestamp column must use the RFC 3339 format, e.g.
      `2012-07-30T10:43:17.123Z`.
    destination: Required. Specifies output location and format.
    entityTypeSpecs: Required. Specifies EntityType grouping Features to read
      values of and settings.
    passThroughFields: When not empty, the specified fields in the
      *_read_instances source will be joined as-is in the output, in addition
      to those fields from the Featurestore Entity. For BigQuery source, the
      type of the pass-through values will be automatically inferred. For CSV
      source, the pass-through values will be passed as opaque bytes.
    startTime: Optional. Excludes Feature values with feature generation
      timestamp before this timestamp. If not set, retrieve oldest values kept
      in Feature Store. Timestamp, if present, must not have higher than
      millisecond precision.
  """
    bigqueryReadInstances = _messages.MessageField('GoogleCloudAiplatformV1BigQuerySource', 1)
    csvReadInstances = _messages.MessageField('GoogleCloudAiplatformV1CsvSource', 2)
    destination = _messages.MessageField('GoogleCloudAiplatformV1FeatureValueDestination', 3)
    entityTypeSpecs = _messages.MessageField('GoogleCloudAiplatformV1BatchReadFeatureValuesRequestEntityTypeSpec', 4, repeated=True)
    passThroughFields = _messages.MessageField('GoogleCloudAiplatformV1BatchReadFeatureValuesRequestPassThroughField', 5, repeated=True)
    startTime = _messages.StringField(6)