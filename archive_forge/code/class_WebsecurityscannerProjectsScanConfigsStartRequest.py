from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsStartRequest(_messages.Message):
    """A WebsecurityscannerProjectsScanConfigsStartRequest object.

  Fields:
    name: Required. The resource name of the ScanConfig to be used. The name
      follows the format of 'projects/{projectId}/scanConfigs/{scanConfigId}'.
    startScanRunRequest: A StartScanRunRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    startScanRunRequest = _messages.MessageField('StartScanRunRequest', 2)