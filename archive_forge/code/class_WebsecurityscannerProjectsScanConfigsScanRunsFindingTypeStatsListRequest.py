from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsScanRunsFindingTypeStatsListRequest(_messages.Message):
    """A
  WebsecurityscannerProjectsScanConfigsScanRunsFindingTypeStatsListRequest
  object.

  Fields:
    parent: Required. The parent resource name, which should be a scan run
      resource name in the format
      'projects/{projectId}/scanConfigs/{scanConfigId}/scanRuns/{scanRunId}'.
  """
    parent = _messages.StringField(1, required=True)