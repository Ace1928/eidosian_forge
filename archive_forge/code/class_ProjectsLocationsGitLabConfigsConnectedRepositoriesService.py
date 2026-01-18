from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsGitLabConfigsConnectedRepositoriesService(base_api.BaseApiService):
    """Service class for the projects_locations_gitLabConfigs_connectedRepositories resource."""
    _NAME = 'projects_locations_gitLabConfigs_connectedRepositories'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsGitLabConfigsConnectedRepositoriesService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Batch connecting GitLab repositories to Cloud Build. This API is experimental.

      Args:
        request: (CloudbuildProjectsLocationsGitLabConfigsConnectedRepositoriesBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gitLabConfigs/{gitLabConfigsId}/connectedRepositories:batchCreate', http_method='POST', method_id='cloudbuild.projects.locations.gitLabConfigs.connectedRepositories.batchCreate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/connectedRepositories:batchCreate', request_field='batchCreateGitLabConnectedRepositoriesRequest', request_type_name='CloudbuildProjectsLocationsGitLabConfigsConnectedRepositoriesBatchCreateRequest', response_type_name='Operation', supports_download=False)