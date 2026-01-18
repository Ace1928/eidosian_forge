from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesGenericArtifactsService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_genericArtifacts resource."""
    _NAME = 'projects_locations_repositories_genericArtifacts'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesGenericArtifactsService, self).__init__(client)
        self._upload_configs = {'Upload': base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=True, resumable_path='/resumable/upload/v1/{+parent}/genericArtifacts:create', simple_multipart=True, simple_path='/upload/v1/{+parent}/genericArtifacts:create')}

    def Upload(self, request, global_params=None, upload=None):
        """Directly uploads a Generic artifact. The returned Operation will complete once the resources are uploaded. Package, Version, and File resources are created based on the uploaded artifact. Uploaded artifacts that conflict with existing resources will raise an ALREADY_EXISTS error.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesGenericArtifactsUploadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (UploadGenericArtifactMediaResponse) The response message.
      """
        config = self.GetMethodConfig('Upload')
        upload_config = self.GetUploadConfig('Upload')
        return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
    Upload.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/genericArtifacts:create', http_method='POST', method_id='artifactregistry.projects.locations.repositories.genericArtifacts.upload', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/genericArtifacts:create', request_field='uploadGenericArtifactRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesGenericArtifactsUploadRequest', response_type_name='UploadGenericArtifactMediaResponse', supports_download=False)