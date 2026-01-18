from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesAptArtifactsService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_aptArtifacts resource."""
    _NAME = 'projects_locations_repositories_aptArtifacts'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesAptArtifactsService, self).__init__(client)
        self._upload_configs = {'Upload': base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=None, resumable_path=None, simple_multipart=True, simple_path='/upload/v1/{+parent}/aptArtifacts:create')}

    def Import(self, request, global_params=None):
        """Imports Apt artifacts. The returned Operation will complete once the resources are imported. Package, Version, and File resources are created based on the imported artifacts. Imported artifacts that conflict with existing resources are ignored.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesAptArtifactsImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/aptArtifacts:import', http_method='POST', method_id='artifactregistry.projects.locations.repositories.aptArtifacts.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/aptArtifacts:import', request_field='importAptArtifactsRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesAptArtifactsImportRequest', response_type_name='Operation', supports_download=False)

    def Upload(self, request, global_params=None, upload=None):
        """Directly uploads an Apt artifact. The returned Operation will complete once the resources are uploaded. Package, Version, and File resources are created based on the imported artifact. Imported artifacts that conflict with existing resources are ignored.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesAptArtifactsUploadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (UploadAptArtifactMediaResponse) The response message.
      """
        config = self.GetMethodConfig('Upload')
        upload_config = self.GetUploadConfig('Upload')
        return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
    Upload.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/aptArtifacts:create', http_method='POST', method_id='artifactregistry.projects.locations.repositories.aptArtifacts.upload', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/aptArtifacts:create', request_field='uploadAptArtifactRequest', request_type_name='ArtifactregistryProjectsLocationsRepositoriesAptArtifactsUploadRequest', response_type_name='UploadAptArtifactMediaResponse', supports_download=False)