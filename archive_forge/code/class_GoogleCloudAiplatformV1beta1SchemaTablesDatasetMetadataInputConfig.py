from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTablesDatasetMetadataInputConfig(_messages.Message):
    """The tables Dataset's data source. The Dataset doesn't store the data
  directly, but only pointer(s) to its data.

  Fields:
    bigquerySource: A
      GoogleCloudAiplatformV1beta1SchemaTablesDatasetMetadataBigQuerySource
      attribute.
    gcsSource: A
      GoogleCloudAiplatformV1beta1SchemaTablesDatasetMetadataGcsSource
      attribute.
  """
    bigquerySource = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTablesDatasetMetadataBigQuerySource', 1)
    gcsSource = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaTablesDatasetMetadataGcsSource', 2)