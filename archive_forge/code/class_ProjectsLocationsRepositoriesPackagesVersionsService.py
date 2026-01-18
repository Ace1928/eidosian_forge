from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesPackagesVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_packages_versions resource."""
    _NAME = 'projects_locations_repositories_packages_versions'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesPackagesVersionsService, self).__init__(client)
        self._upload_configs = {}

    def BatchDelete(self, request, global_params=None):
        """Deletes multiple versions across a repository. The returned operation will complete once the versions have been deleted.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsBatchDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('BatchDelete')
        return self._RunMethod(config, request, global_params=global_params)
    BatchDelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/packages/{packagesId}/versions:batchDelete', http_method='POST', method_id='artifactregistry.projects.locations.repositories.packages.versions.batchDelete', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/versions:batchDelete', request_field='batchDeleteVersionsRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsBatchDeleteRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a version and all of its content. The returned operation will complete once the version has been deleted.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/packages/{packagesId}/versions/{versionsId}', http_method='DELETE', method_id='artifactregistry.projects.locations.repositories.packages.versions.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a version.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Version) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/packages/{packagesId}/versions/{versionsId}', http_method='GET', method_id='artifactregistry.projects.locations.repositories.packages.versions.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsGetRequest', response_type_name='Version', supports_download=False)

    def List(self, request, global_params=None):
        """Lists versions.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/packages/{packagesId}/versions', http_method='GET', method_id='artifactregistry.projects.locations.repositories.packages.versions.list', ordered_params=['parent'], path_params=['parent'], query_params=['orderBy', 'pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/versions', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsListRequest', response_type_name='ListVersionsResponse', supports_download=False)