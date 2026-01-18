from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vision.v1 import vision_v1_messages as messages
class ProjectsLocationsFilesService(base_api.BaseApiService):
    """Service class for the projects_locations_files resource."""
    _NAME = 'projects_locations_files'

    def __init__(self, client):
        super(VisionV1.ProjectsLocationsFilesService, self).__init__(client)
        self._upload_configs = {}

    def Annotate(self, request, global_params=None):
        """Service that performs image detection and annotation for a batch of files. Now only "application/pdf", "image/tiff" and "image/gif" are supported. This service will extract at most 5 (customers can specify which 5 in AnnotateFileRequest.pages) frames (gif) or pages (pdf or tiff) from each file provided and perform detection and annotation for each image extracted.

      Args:
        request: (BatchAnnotateFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchAnnotateFilesResponse) The response message.
      """
        config = self.GetMethodConfig('Annotate')
        return self._RunMethod(config, request, global_params=global_params)
    Annotate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/files:annotate', http_method='POST', method_id='vision.projects.locations.files.annotate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/files:annotate', request_field='<request>', request_type_name='BatchAnnotateFilesRequest', response_type_name='BatchAnnotateFilesResponse', supports_download=False)

    def AsyncBatchAnnotate(self, request, global_params=None):
        """Run asynchronous image detection and annotation for a list of generic files, such as PDF files, which may contain multiple pages and multiple images per page. Progress and results can be retrieved through the `google.longrunning.Operations` interface. `Operation.metadata` contains `OperationMetadata` (metadata). `Operation.response` contains `AsyncBatchAnnotateFilesResponse` (results).

      Args:
        request: (AsyncBatchAnnotateFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AsyncBatchAnnotate')
        return self._RunMethod(config, request, global_params=global_params)
    AsyncBatchAnnotate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/files:asyncBatchAnnotate', http_method='POST', method_id='vision.projects.locations.files.asyncBatchAnnotate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/files:asyncBatchAnnotate', request_field='<request>', request_type_name='AsyncBatchAnnotateFilesRequest', response_type_name='Operation', supports_download=False)