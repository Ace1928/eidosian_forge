from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ProjectsGroupsMembersService(base_api.BaseApiService):
    """Service class for the projects_groups_members resource."""
    _NAME = 'projects_groups_members'

    def __init__(self, client):
        super(MonitoringV3.ProjectsGroupsMembersService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the monitored resources that are members of a group.

      Args:
        request: (MonitoringProjectsGroupsMembersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGroupMembersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/groups/{groupsId}/members', http_method='GET', method_id='monitoring.projects.groups.members.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'interval_endTime', 'interval_startTime', 'pageSize', 'pageToken'], relative_path='v3/{+name}/members', request_field='', request_type_name='MonitoringProjectsGroupsMembersListRequest', response_type_name='ListGroupMembersResponse', supports_download=False)