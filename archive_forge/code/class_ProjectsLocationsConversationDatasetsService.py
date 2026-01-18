from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsConversationDatasetsService(base_api.BaseApiService):
    """Service class for the projects_locations_conversationDatasets resource."""
    _NAME = 'projects_locations_conversationDatasets'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsConversationDatasetsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new conversation dataset. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: CreateConversationDatasetOperationMetadata - `response`: ConversationDataset.

      Args:
        request: (DialogflowProjectsLocationsConversationDatasetsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversationDatasets', http_method='POST', method_id='dialogflow.projects.locations.conversationDatasets.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/conversationDatasets', request_field='googleCloudDialogflowV2ConversationDataset', request_type_name='DialogflowProjectsLocationsConversationDatasetsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified conversation dataset. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: DeleteConversationDatasetOperationMetadata - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty).

      Args:
        request: (DialogflowProjectsLocationsConversationDatasetsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversationDatasets/{conversationDatasetsId}', http_method='DELETE', method_id='dialogflow.projects.locations.conversationDatasets.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsConversationDatasetsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified conversation dataset.

      Args:
        request: (DialogflowProjectsLocationsConversationDatasetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ConversationDataset) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversationDatasets/{conversationDatasetsId}', http_method='GET', method_id='dialogflow.projects.locations.conversationDatasets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsConversationDatasetsGetRequest', response_type_name='GoogleCloudDialogflowV2ConversationDataset', supports_download=False)

    def ImportConversationData(self, request, global_params=None):
        """Import data into the specified conversation dataset. Note that it is not allowed to import data to a conversation dataset that already has data in it. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: ImportConversationDataOperationMetadata - `response`: ImportConversationDataOperationResponse.

      Args:
        request: (DialogflowProjectsLocationsConversationDatasetsImportConversationDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ImportConversationData')
        return self._RunMethod(config, request, global_params=global_params)
    ImportConversationData.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversationDatasets/{conversationDatasetsId}:importConversationData', http_method='POST', method_id='dialogflow.projects.locations.conversationDatasets.importConversationData', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:importConversationData', request_field='googleCloudDialogflowV2ImportConversationDataRequest', request_type_name='DialogflowProjectsLocationsConversationDatasetsImportConversationDataRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all conversation datasets in the specified project and location.

      Args:
        request: (DialogflowProjectsLocationsConversationDatasetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListConversationDatasetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversationDatasets', http_method='GET', method_id='dialogflow.projects.locations.conversationDatasets.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/conversationDatasets', request_field='', request_type_name='DialogflowProjectsLocationsConversationDatasetsListRequest', response_type_name='GoogleCloudDialogflowV2ListConversationDatasetsResponse', supports_download=False)