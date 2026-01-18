from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileSpecPostScanActionsBigQueryExport(_messages.Message):
    """The configuration of BigQuery export post scan action.

  Fields:
    resultsTable: Optional. The BigQuery table to export DataProfileScan
      results to. Format: //bigquery.googleapis.com/projects/PROJECT_ID/datase
      ts/DATASET_ID/tables/TABLE_ID
  """
    resultsTable = _messages.StringField(1)