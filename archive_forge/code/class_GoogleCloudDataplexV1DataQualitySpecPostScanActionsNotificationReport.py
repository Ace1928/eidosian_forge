from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualitySpecPostScanActionsNotificationReport(_messages.Message):
    """The configuration of notification report post scan action.

  Fields:
    jobEndTrigger: Optional. If set, report will be sent when a scan job ends.
    jobFailureTrigger: Optional. If set, report will be sent when a scan job
      fails.
    recipients: Required. The recipients who will receive the notification
      report.
    scoreThresholdTrigger: Optional. If set, report will be sent when score
      threshold is met.
  """
    jobEndTrigger = _messages.MessageField('GoogleCloudDataplexV1DataQualitySpecPostScanActionsJobEndTrigger', 1)
    jobFailureTrigger = _messages.MessageField('GoogleCloudDataplexV1DataQualitySpecPostScanActionsJobFailureTrigger', 2)
    recipients = _messages.MessageField('GoogleCloudDataplexV1DataQualitySpecPostScanActionsRecipients', 3)
    scoreThresholdTrigger = _messages.MessageField('GoogleCloudDataplexV1DataQualitySpecPostScanActionsScoreThresholdTrigger', 4)