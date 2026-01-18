from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsVpcFlowLogsConfigsPatchRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsVpcFlowLogsConfigsPatchRequest
  object.

  Fields:
    name: Identifier. Unique name of the configuration using the form:
      `projects/{project_id}/locations/global/vpcFlowLogs/{vpc_flow_log}`
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field.
    vpcFlowLogsConfig: A VpcFlowLogsConfig resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    vpcFlowLogsConfig = _messages.MessageField('VpcFlowLogsConfig', 3)