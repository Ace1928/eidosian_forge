from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsVpcFlowLogsConfigsListRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsVpcFlowLogsConfigsListRequest
  object.

  Fields:
    filter: Optional. Lists the `VpcFlowLogsConfig`s that match the filter
      expression. A filter expression filters the resources listed in the
      response. The expression must be of the form ` ` where operators: `<`,
      `>`, `<=`, `>=`, `!=`, `=`, `:` are supported (colon `:` represents a
      HAS operator which is roughly synonymous with equality). can refer to a
      proto or JSON field, or a synthetic field. Field names can be camelCase
      or snake_case. Examples: - Filter by name: name =
      "projects/proj-1/locations/global/vpcFlowLogsConfigs/config-1 - Filter
      by target resource: - Configurations at the VPC network level
      target_resource.network:* - Configurations for a VPC network called
      `vpc-1` target_resource.network = vpc-1
    orderBy: Optional. Field to use to sort the list.
    pageSize: Optional. Number of `VpcFlowLogsConfig`s to return.
    pageToken: Optional. Page token from an earlier query, as returned in
      `next_page_token`.
    parent: Required. The parent resource of the VpcFlowLogsConfig:
      `projects/{project_id}/locations/global`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)