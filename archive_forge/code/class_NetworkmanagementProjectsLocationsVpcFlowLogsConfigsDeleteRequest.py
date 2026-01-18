from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsVpcFlowLogsConfigsDeleteRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsVpcFlowLogsConfigsDeleteRequest
  object.

  Fields:
    name: Required. VpcFlowLogsConfig name using the form: `projects/{project_
      id}/locations/global/vpcFlowLogsConfigs/{vpc_flow_logs_config}`
  """
    name = _messages.StringField(1, required=True)