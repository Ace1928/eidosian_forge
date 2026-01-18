from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesNpmPackagesService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_npmPackages resource."""
    _NAME = 'projects_locations_repositories_npmPackages'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesNpmPackagesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a npm package.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesNpmPackagesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NpmPackage) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/npmPackages/{npmPackagesId}', http_method='GET', method_id='artifactregistry.projects.locations.repositories.npmPackages.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesNpmPackagesGetRequest', response_type_name='NpmPackage', supports_download=False)

    def List(self, request, global_params=None):
        """Lists npm packages.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesNpmPackagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNpmPackagesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/npmPackages', http_method='GET', method_id='artifactregistry.projects.locations.repositories.npmPackages.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/npmPackages', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesNpmPackagesListRequest', response_type_name='ListNpmPackagesResponse', supports_download=False)