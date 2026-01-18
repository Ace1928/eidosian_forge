from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsScanRunsStopRequest(_messages.Message):
    """A WebsecurityscannerProjectsScanConfigsScanRunsStopRequest object.

  Fields:
    name: Required. The resource name of the ScanRun to be stopped. The name
      follows the format of
      'projects/{projectId}/scanConfigs/{scanConfigId}/scanRuns/{scanRunId}'.
    stopScanRunRequest: A StopScanRunRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    stopScanRunRequest = _messages.MessageField('StopScanRunRequest', 2)