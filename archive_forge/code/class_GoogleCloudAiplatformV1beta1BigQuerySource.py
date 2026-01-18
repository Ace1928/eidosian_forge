from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BigQuerySource(_messages.Message):
    """The BigQuery location for the input content.

  Fields:
    inputUri: Required. BigQuery URI to a table, up to 2000 characters long.
      Accepted forms: * BigQuery path. For example:
      `bq://projectId.bqDatasetId.bqTableId`.
  """
    inputUri = _messages.StringField(1)