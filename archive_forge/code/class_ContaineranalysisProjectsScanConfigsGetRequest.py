from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsScanConfigsGetRequest(_messages.Message):
    """A ContaineranalysisProjectsScanConfigsGetRequest object.

  Fields:
    name: The name of the ScanConfig in the form
      projects/{project_id}/scanConfigs/{scan_config_id}
  """
    name = _messages.StringField(1, required=True)