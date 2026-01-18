from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesKfpArtifactsService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_kfpArtifacts resource."""
    _NAME = 'projects_locations_repositories_kfpArtifacts'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesKfpArtifactsService, self).__init__(client)
        self._upload_configs = {'Upload': base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=None, resumable_path=None, simple_multipart=True, simple_path='/upload/v1/{+parent}/kfpArtifacts:create')}

    def Upload(self, request, global_params=None, upload=None):
        """Directly uploads a KFP artifact. The returned Operation will complete once the resource is uploaded. Package, Version, and File resources will be created based on the uploaded artifact. Uploaded artifacts that conflict with existing resources will be overwritten.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesKfpArtifactsUploadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (UploadKfpArtifactMediaResponse) The response message.
      """
        config = self.GetMethodConfig('Upload')
        upload_config = self.GetUploadConfig('Upload')
        return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
    Upload.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/kfpArtifacts:create', http_method='POST', method_id='artifactregistry.projects.locations.repositories.kfpArtifacts.upload', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/kfpArtifacts:create', request_field='uploadKfpArtifactRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesKfpArtifactsUploadRequest', response_type_name='UploadKfpArtifactMediaResponse', supports_download=False)