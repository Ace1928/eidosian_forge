from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualitySpecPostScanActions(_messages.Message):
    """The configuration of post scan actions of DataQualityScan.

  Fields:
    bigqueryExport: Optional. If set, results will be exported to the provided
      BigQuery table.
    notificationReport: Optional. If set, results will be sent to the provided
      notification receipts upon triggers.
  """
    bigqueryExport = _messages.MessageField('GoogleCloudDataplexV1DataQualitySpecPostScanActionsBigQueryExport', 1)
    notificationReport = _messages.MessageField('GoogleCloudDataplexV1DataQualitySpecPostScanActionsNotificationReport', 2)