from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2BigQueryTable(_messages.Message):
    """Message defining the location of a BigQuery table. A table is uniquely
  identified by its project_id, dataset_id, and table_name. Within a query a
  table is often referenced with a string in the format of: `:.` or `..`.

  Fields:
    datasetId: Dataset ID of the table.
    projectId: The Google Cloud Platform project ID of the project containing
      the table. If omitted, project ID is inferred from the API call.
    tableId: Name of the table.
  """
    datasetId = _messages.StringField(1)
    projectId = _messages.StringField(2)
    tableId = _messages.StringField(3)