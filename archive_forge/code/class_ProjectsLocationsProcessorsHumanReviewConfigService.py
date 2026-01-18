from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
class ProjectsLocationsProcessorsHumanReviewConfigService(base_api.BaseApiService):
    """Service class for the projects_locations_processors_humanReviewConfig resource."""
    _NAME = 'projects_locations_processors_humanReviewConfig'

    def __init__(self, client):
        super(DocumentaiV1.ProjectsLocationsProcessorsHumanReviewConfigService, self).__init__(client)
        self._upload_configs = {}

    def ReviewDocument(self, request, global_params=None):
        """Send a document for Human Review. The input document should be processed by the specified processor.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsHumanReviewConfigReviewDocumentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ReviewDocument')
        return self._RunMethod(config, request, global_params=global_params)
    ReviewDocument.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/humanReviewConfig:reviewDocument', http_method='POST', method_id='documentai.projects.locations.processors.humanReviewConfig.reviewDocument', ordered_params=['humanReviewConfig'], path_params=['humanReviewConfig'], query_params=[], relative_path='v1/{+humanReviewConfig}:reviewDocument', request_field='googleCloudDocumentaiV1ReviewDocumentRequest', request_type_name='DocumentaiProjectsLocationsProcessorsHumanReviewConfigReviewDocumentRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)