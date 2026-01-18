from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouderrorreporting.v1beta1 import clouderrorreporting_v1beta1_messages as messages
class ProjectsGroupsService(base_api.BaseApiService):
    """Service class for the projects_groups resource."""
    _NAME = 'projects_groups'

    def __init__(self, client):
        super(ClouderrorreportingV1beta1.ProjectsGroupsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get the specified group.

      Args:
        request: (ClouderrorreportingProjectsGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ErrorGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/groups/{groupsId}', http_method='GET', method_id='clouderrorreporting.projects.groups.get', ordered_params=['groupName'], path_params=['groupName'], query_params=[], relative_path='v1beta1/{+groupName}', request_field='', request_type_name='ClouderrorreportingProjectsGroupsGetRequest', response_type_name='ErrorGroup', supports_download=False)

    def Update(self, request, global_params=None):
        """Replace the data for the specified group. Fails if the group does not exist.

      Args:
        request: (ErrorGroup) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ErrorGroup) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/groups/{groupsId}', http_method='PUT', method_id='clouderrorreporting.projects.groups.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='<request>', request_type_name='ErrorGroup', response_type_name='ErrorGroup', supports_download=False)