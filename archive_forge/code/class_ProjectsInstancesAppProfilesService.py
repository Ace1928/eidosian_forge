from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
class ProjectsInstancesAppProfilesService(base_api.BaseApiService):
    """Service class for the projects_instances_appProfiles resource."""
    _NAME = 'projects_instances_appProfiles'

    def __init__(self, client):
        super(BigtableadminV2.ProjectsInstancesAppProfilesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an app profile within an instance.

      Args:
        request: (BigtableadminProjectsInstancesAppProfilesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AppProfile) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/appProfiles', http_method='POST', method_id='bigtableadmin.projects.instances.appProfiles.create', ordered_params=['parent'], path_params=['parent'], query_params=['appProfileId', 'ignoreWarnings'], relative_path='v2/{+parent}/appProfiles', request_field='appProfile', request_type_name='BigtableadminProjectsInstancesAppProfilesCreateRequest', response_type_name='AppProfile', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an app profile from an instance.

      Args:
        request: (BigtableadminProjectsInstancesAppProfilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/appProfiles/{appProfilesId}', http_method='DELETE', method_id='bigtableadmin.projects.instances.appProfiles.delete', ordered_params=['name'], path_params=['name'], query_params=['ignoreWarnings'], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesAppProfilesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about an app profile.

      Args:
        request: (BigtableadminProjectsInstancesAppProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AppProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/appProfiles/{appProfilesId}', http_method='GET', method_id='bigtableadmin.projects.instances.appProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesAppProfilesGetRequest', response_type_name='AppProfile', supports_download=False)

    def List(self, request, global_params=None):
        """Lists information about app profiles in an instance.

      Args:
        request: (BigtableadminProjectsInstancesAppProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAppProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/appProfiles', http_method='GET', method_id='bigtableadmin.projects.instances.appProfiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/appProfiles', request_field='', request_type_name='BigtableadminProjectsInstancesAppProfilesListRequest', response_type_name='ListAppProfilesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an app profile within an instance.

      Args:
        request: (BigtableadminProjectsInstancesAppProfilesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/appProfiles/{appProfilesId}', http_method='PATCH', method_id='bigtableadmin.projects.instances.appProfiles.patch', ordered_params=['name'], path_params=['name'], query_params=['ignoreWarnings', 'updateMask'], relative_path='v2/{+name}', request_field='appProfile', request_type_name='BigtableadminProjectsInstancesAppProfilesPatchRequest', response_type_name='Operation', supports_download=False)