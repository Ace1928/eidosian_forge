from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsReasoningEnginesService(base_api.BaseApiService):
    """Service class for the projects_locations_reasoningEngines resource."""
    _NAME = 'projects_locations_reasoningEngines'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsReasoningEnginesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a reasoning engine.

      Args:
        request: (AiplatformProjectsLocationsReasoningEnginesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/reasoningEngines', http_method='POST', method_id='aiplatform.projects.locations.reasoningEngines.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta1/{+parent}/reasoningEngines', request_field='googleCloudAiplatformV1beta1ReasoningEngine', request_type_name='AiplatformProjectsLocationsReasoningEnginesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a reasoning engine.

      Args:
        request: (AiplatformProjectsLocationsReasoningEnginesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/reasoningEngines/{reasoningEnginesId}', http_method='DELETE', method_id='aiplatform.projects.locations.reasoningEngines.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsReasoningEnginesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a reasoning engine.

      Args:
        request: (AiplatformProjectsLocationsReasoningEnginesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ReasoningEngine) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/reasoningEngines/{reasoningEnginesId}', http_method='GET', method_id='aiplatform.projects.locations.reasoningEngines.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsReasoningEnginesGetRequest', response_type_name='GoogleCloudAiplatformV1beta1ReasoningEngine', supports_download=False)

    def List(self, request, global_params=None):
        """Lists reasoning engines in a location.

      Args:
        request: (AiplatformProjectsLocationsReasoningEnginesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ListReasoningEnginesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/reasoningEngines', http_method='GET', method_id='aiplatform.projects.locations.reasoningEngines.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/reasoningEngines', request_field='', request_type_name='AiplatformProjectsLocationsReasoningEnginesListRequest', response_type_name='GoogleCloudAiplatformV1beta1ListReasoningEnginesResponse', supports_download=False)

    def Query(self, request, global_params=None):
        """Queries using a reasoning engine.

      Args:
        request: (AiplatformProjectsLocationsReasoningEnginesQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1QueryReasoningEngineResponse) The response message.
      """
        config = self.GetMethodConfig('Query')
        return self._RunMethod(config, request, global_params=global_params)
    Query.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/reasoningEngines/{reasoningEnginesId}:query', http_method='POST', method_id='aiplatform.projects.locations.reasoningEngines.query', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}:query', request_field='googleCloudAiplatformV1beta1QueryReasoningEngineRequest', request_type_name='AiplatformProjectsLocationsReasoningEnginesQueryRequest', response_type_name='GoogleCloudAiplatformV1beta1QueryReasoningEngineResponse', supports_download=False)