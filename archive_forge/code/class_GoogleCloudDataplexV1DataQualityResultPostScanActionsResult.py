from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityResultPostScanActionsResult(_messages.Message):
    """The result of post scan actions of DataQualityScan job.

  Fields:
    bigqueryExportResult: Output only. The result of BigQuery export post scan
      action.
  """
    bigqueryExportResult = _messages.MessageField('GoogleCloudDataplexV1DataQualityResultPostScanActionsResultBigQueryExportResult', 1)