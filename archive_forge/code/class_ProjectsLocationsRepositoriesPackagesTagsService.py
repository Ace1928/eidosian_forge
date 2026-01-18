from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesPackagesTagsService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_packages_tags resource."""
    _NAME = 'projects_locations_repositories_packages_tags'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesPackagesTagsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a tag.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPackagesTagsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Tag) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/packages/{packagesId}/tags', http_method='POST', method_id='artifactregistry.projects.locations.repositories.packages.tags.create', ordered_params=['parent'], path_params=['parent'], query_params=['tagId'], relative_path='v1/{+parent}/tags', request_field='tag', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPackagesTagsCreateRequest', response_type_name='Tag', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a tag.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPackagesTagsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/packages/{packagesId}/tags/{tagsId}', http_method='DELETE', method_id='artifactregistry.projects.locations.repositories.packages.tags.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPackagesTagsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a tag.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPackagesTagsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Tag) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/packages/{packagesId}/tags/{tagsId}', http_method='GET', method_id='artifactregistry.projects.locations.repositories.packages.tags.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPackagesTagsGetRequest', response_type_name='Tag', supports_download=False)

    def List(self, request, global_params=None):
        """Lists tags.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPackagesTagsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTagsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/packages/{packagesId}/tags', http_method='GET', method_id='artifactregistry.projects.locations.repositories.packages.tags.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/tags', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPackagesTagsListRequest', response_type_name='ListTagsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a tag.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesPackagesTagsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Tag) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/packages/{packagesId}/tags/{tagsId}', http_method='PATCH', method_id='artifactregistry.projects.locations.repositories.packages.tags.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='tag', request_type_name='ArtifactregistryProjectsLocationsRepositoriesPackagesTagsPatchRequest', response_type_name='Tag', supports_download=False)