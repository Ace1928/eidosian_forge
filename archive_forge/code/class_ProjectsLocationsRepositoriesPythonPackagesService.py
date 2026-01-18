from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesPythonPackagesService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_pythonPackages resource."""
    _NAME = 'projects_locations_repositories_pythonPackages'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesPythonPackagesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a python package.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPythonPackagesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PythonPackage) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/pythonPackages/{pythonPackagesId}', http_method='GET', method_id='artifactregistry.projects.locations.repositories.pythonPackages.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPythonPackagesGetRequest', response_type_name='PythonPackage', supports_download=False)

    def List(self, request, global_params=None):
        """Lists python packages.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPythonPackagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPythonPackagesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/pythonPackages', http_method='GET', method_id='artifactregistry.projects.locations.repositories.pythonPackages.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/pythonPackages', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPythonPackagesListRequest', response_type_name='ListPythonPackagesResponse', supports_download=False)