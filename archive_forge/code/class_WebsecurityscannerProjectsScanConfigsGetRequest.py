from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WebsecurityscannerProjectsScanConfigsGetRequest(_messages.Message):
    """A WebsecurityscannerProjectsScanConfigsGetRequest object.

  Fields:
    name: Required. The resource name of the ScanConfig to be returned. The
      name follows the format of
      'projects/{projectId}/scanConfigs/{scanConfigId}'.
  """
    name = _messages.StringField(1, required=True)