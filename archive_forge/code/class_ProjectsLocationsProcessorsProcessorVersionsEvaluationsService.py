from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.documentai.v1 import documentai_v1_messages as messages
class ProjectsLocationsProcessorsProcessorVersionsEvaluationsService(base_api.BaseApiService):
    """Service class for the projects_locations_processors_processorVersions_evaluations resource."""
    _NAME = 'projects_locations_processors_processorVersions_evaluations'

    def __init__(self, client):
        super(DocumentaiV1.ProjectsLocationsProcessorsProcessorVersionsEvaluationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves a specific evaluation.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1Evaluation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions/{processorVersionsId}/evaluations/{evaluationsId}', http_method='GET', method_id='documentai.projects.locations.processors.processorVersions.evaluations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluationsGetRequest', response_type_name='GoogleCloudDocumentaiV1Evaluation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a set of evaluations for a given processor version.

      Args:
        request: (DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDocumentaiV1ListEvaluationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/processors/{processorsId}/processorVersions/{processorVersionsId}/evaluations', http_method='GET', method_id='documentai.projects.locations.processors.processorVersions.evaluations.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/evaluations', request_field='', request_type_name='DocumentaiProjectsLocationsProcessorsProcessorVersionsEvaluationsListRequest', response_type_name='GoogleCloudDocumentaiV1ListEvaluationsResponse', supports_download=False)