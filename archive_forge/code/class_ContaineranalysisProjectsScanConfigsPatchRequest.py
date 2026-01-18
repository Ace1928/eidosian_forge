from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsScanConfigsPatchRequest(_messages.Message):
    """A ContaineranalysisProjectsScanConfigsPatchRequest object.

  Fields:
    name: The scan config to update of the form
      projects/{project_id}/scanConfigs/{scan_config_id}.
    scanConfig: A ScanConfig resource to be passed as the request body.
    updateMask: The fields to update.
  """
    name = _messages.StringField(1, required=True)
    scanConfig = _messages.MessageField('ScanConfig', 2)
    updateMask = _messages.StringField(3)