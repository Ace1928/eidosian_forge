from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsGithubEnterpriseConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_githubEnterpriseConfigs resource."""
    _NAME = 'projects_locations_githubEnterpriseConfigs'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsGithubEnterpriseConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an association between a GCP project and a GitHub Enterprise server.

      Args:
        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs', http_method='POST', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['gheConfigId', 'projectId'], relative_path='v1/{+parent}/githubEnterpriseConfigs', request_field='gitHubEnterpriseConfig', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete an association between a GCP project and a GitHub Enterprise server.

      Args:
        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}', http_method='DELETE', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=['configId', 'projectId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve a GitHubEnterpriseConfig.

      Args:
        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GitHubEnterpriseConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}', http_method='GET', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.get', ordered_params=['name'], path_params=['name'], query_params=['configId', 'projectId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsGetRequest', response_type_name='GitHubEnterpriseConfig', supports_download=False)

    def GetApp(self, request, global_params=None):
        """Get the GitHub App associated with a GitHub Enterprise Config. Uses the GitHub API: https://developer.github.com/enterprise/2.21/v3/apps/#get-an-app This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsGetAppRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GitHubEnterpriseApp) The response message.
      """
        config = self.GetMethodConfig('GetApp')
        return self._RunMethod(config, request, global_params=global_params)
    GetApp.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}/app', http_method='GET', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.getApp', ordered_params=['enterpriseConfigResource'], path_params=['enterpriseConfigResource'], query_params=[], relative_path='v1/{+enterpriseConfigResource}/app', request_field='', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsGetAppRequest', response_type_name='GitHubEnterpriseApp', supports_download=False)

    def List(self, request, global_params=None):
        """List all GitHubEnterpriseConfigs for a given project.

      Args:
        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGithubEnterpriseConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs', http_method='GET', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['projectId'], relative_path='v1/{+parent}/githubEnterpriseConfigs', request_field='', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsListRequest', response_type_name='ListGithubEnterpriseConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update an association between a GCP project and a GitHub Enterprise server.

      Args:
        request: (CloudbuildProjectsLocationsGithubEnterpriseConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/githubEnterpriseConfigs/{githubEnterpriseConfigsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.githubEnterpriseConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='gitHubEnterpriseConfig', request_type_name='CloudbuildProjectsLocationsGithubEnterpriseConfigsPatchRequest', response_type_name='Operation', supports_download=False)