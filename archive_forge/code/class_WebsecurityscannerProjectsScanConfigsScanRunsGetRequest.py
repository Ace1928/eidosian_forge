from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsScanRunsGetRequest(_messages.Message):
    """A WebsecurityscannerProjectsScanConfigsScanRunsGetRequest object.

  Fields:
    name: Required. The resource name of the ScanRun to be returned. The name
      follows the format of
      'projects/{projectId}/scanConfigs/{scanConfigId}/scanRuns/{scanRunId}'.
  """
    name = _messages.StringField(1, required=True)