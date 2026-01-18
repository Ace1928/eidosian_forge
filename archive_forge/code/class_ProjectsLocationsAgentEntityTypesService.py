from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsAgentEntityTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_agent_entityTypes resource."""
    _NAME = 'projects_locations_agent_entityTypes'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsAgentEntityTypesService, self).__init__(client)
        self._upload_configs = {}

    def BatchDelete(self, request, global_params=None):
        """Deletes entity types in the specified agent. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentEntityTypesBatchDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchDelete')
        return self._RunMethod(config, request, global_params=global_params)
    BatchDelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/entityTypes:batchDelete', http_method='POST', method_id='dialogflow.projects.locations.agent.entityTypes.batchDelete', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/entityTypes:batchDelete', request_field='googleCloudDialogflowV2BatchDeleteEntityTypesRequest', request_type_name='DialogflowProjectsLocationsAgentEntityTypesBatchDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def BatchUpdate(self, request, global_params=None):
        """Updates/Creates multiple entity types in the specified agent. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: BatchUpdateEntityTypesResponse Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentEntityTypesBatchUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchUpdate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchUpdate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/entityTypes:batchUpdate', http_method='POST', method_id='dialogflow.projects.locations.agent.entityTypes.batchUpdate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/entityTypes:batchUpdate', request_field='googleCloudDialogflowV2BatchUpdateEntityTypesRequest', request_type_name='DialogflowProjectsLocationsAgentEntityTypesBatchUpdateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates an entity type in the specified agent. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentEntityTypesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2EntityType) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/entityTypes', http_method='POST', method_id='dialogflow.projects.locations.agent.entityTypes.create', ordered_params=['parent'], path_params=['parent'], query_params=['languageCode'], relative_path='v2/{+parent}/entityTypes', request_field='googleCloudDialogflowV2EntityType', request_type_name='DialogflowProjectsLocationsAgentEntityTypesCreateRequest', response_type_name='GoogleCloudDialogflowV2EntityType', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified entity type. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentEntityTypesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/entityTypes/{entityTypesId}', http_method='DELETE', method_id='dialogflow.projects.locations.agent.entityTypes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentEntityTypesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified entity type.

      Args:
        request: (DialogflowProjectsLocationsAgentEntityTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2EntityType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/entityTypes/{entityTypesId}', http_method='GET', method_id='dialogflow.projects.locations.agent.entityTypes.get', ordered_params=['name'], path_params=['name'], query_params=['languageCode'], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentEntityTypesGetRequest', response_type_name='GoogleCloudDialogflowV2EntityType', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all entity types in the specified agent.

      Args:
        request: (DialogflowProjectsLocationsAgentEntityTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListEntityTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/entityTypes', http_method='GET', method_id='dialogflow.projects.locations.agent.entityTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['languageCode', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/entityTypes', request_field='', request_type_name='DialogflowProjectsLocationsAgentEntityTypesListRequest', response_type_name='GoogleCloudDialogflowV2ListEntityTypesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified entity type. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentEntityTypesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2EntityType) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/entityTypes/{entityTypesId}', http_method='PATCH', method_id='dialogflow.projects.locations.agent.entityTypes.patch', ordered_params=['name'], path_params=['name'], query_params=['languageCode', 'updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2EntityType', request_type_name='DialogflowProjectsLocationsAgentEntityTypesPatchRequest', response_type_name='GoogleCloudDialogflowV2EntityType', supports_download=False)