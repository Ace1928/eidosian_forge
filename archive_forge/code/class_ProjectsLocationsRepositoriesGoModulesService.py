from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesGoModulesService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_goModules resource."""
    _NAME = 'projects_locations_repositories_goModules'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesGoModulesService, self).__init__(client)
        self._upload_configs = {'Upload': base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=None, resumable_path=None, simple_multipart=True, simple_path='/upload/v1/{+parent}/goModules:create')}

    def Upload(self, request, global_params=None, upload=None):
        """Directly uploads a Go module. The returned Operation will complete once the Go module is uploaded. Package, Version, and File resources are created based on the uploaded Go module.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesGoModulesUploadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (UploadGoModuleMediaResponse) The response message.
      """
        config = self.GetMethodConfig('Upload')
        upload_config = self.GetUploadConfig('Upload')
        return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
    Upload.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/goModules:create', http_method='POST', method_id='artifactregistry.projects.locations.repositories.goModules.upload', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/goModules:create', request_field='uploadGoModuleRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesGoModulesUploadRequest', response_type_name='UploadGoModuleMediaResponse', supports_download=False)