from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTablesDatasetMetadataBigQuerySource(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTablesDatasetMetadataBigQuerySource
  object.

  Fields:
    uri: The URI of a BigQuery table. e.g.
      bq://projectId.bqDatasetId.bqTableId
  """
    uri = _messages.StringField(1)