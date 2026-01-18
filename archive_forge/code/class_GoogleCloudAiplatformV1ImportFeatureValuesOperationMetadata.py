from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ImportFeatureValuesOperationMetadata(_messages.Message):
    """Details of operations that perform import Feature values.

  Fields:
    blockingOperationIds: List of ImportFeatureValues operations running under
      a single EntityType that are blocking this operation.
    genericMetadata: Operation metadata for Featurestore import Feature
      values.
    importedEntityCount: Number of entities that have been imported by the
      operation.
    importedFeatureValueCount: Number of Feature values that have been
      imported by the operation.
    invalidRowCount: The number of rows in input source that weren't imported
      due to either * Not having any featureValues. * Having a null entityId.
      * Having a null timestamp. * Not being parsable (applicable for CSV
      sources).
    sourceUris: The source URI from where Feature values are imported.
    timestampOutsideRetentionRowsCount: The number rows that weren't ingested
      due to having timestamps outside the retention boundary.
  """
    blockingOperationIds = _messages.IntegerField(1, repeated=True)
    genericMetadata = _messages.MessageField('GoogleCloudAiplatformV1GenericOperationMetadata', 2)
    importedEntityCount = _messages.IntegerField(3)
    importedFeatureValueCount = _messages.IntegerField(4)
    invalidRowCount = _messages.IntegerField(5)
    sourceUris = _messages.StringField(6, repeated=True)
    timestampOutsideRetentionRowsCount = _messages.IntegerField(7)