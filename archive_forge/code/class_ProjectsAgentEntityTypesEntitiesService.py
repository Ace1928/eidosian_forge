from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsAgentEntityTypesEntitiesService(base_api.BaseApiService):
    """Service class for the projects_agent_entityTypes_entities resource."""
    _NAME = 'projects_agent_entityTypes_entities'

    def __init__(self, client):
        super(DialogflowV2.ProjectsAgentEntityTypesEntitiesService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Creates multiple new entities in the specified entity type. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsAgentEntityTypesEntitiesBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/entityTypes/{entityTypesId}/entities:batchCreate', http_method='POST', method_id='dialogflow.projects.agent.entityTypes.entities.batchCreate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/entities:batchCreate', request_field='googleCloudDialogflowV2BatchCreateEntitiesRequest', request_type_name='DialogflowProjectsAgentEntityTypesEntitiesBatchCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def BatchDelete(self, request, global_params=None):
        """Deletes entities in the specified entity type. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsAgentEntityTypesEntitiesBatchDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchDelete')
        return self._RunMethod(config, request, global_params=global_params)
    BatchDelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/entityTypes/{entityTypesId}/entities:batchDelete', http_method='POST', method_id='dialogflow.projects.agent.entityTypes.entities.batchDelete', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/entities:batchDelete', request_field='googleCloudDialogflowV2BatchDeleteEntitiesRequest', request_type_name='DialogflowProjectsAgentEntityTypesEntitiesBatchDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def BatchUpdate(self, request, global_params=None):
        """Updates or creates multiple entities in the specified entity type. This method does not affect entities in the entity type that aren't explicitly specified in the request. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training). .

      Args:
        request: (DialogflowProjectsAgentEntityTypesEntitiesBatchUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchUpdate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchUpdate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/entityTypes/{entityTypesId}/entities:batchUpdate', http_method='POST', method_id='dialogflow.projects.agent.entityTypes.entities.batchUpdate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/entities:batchUpdate', request_field='googleCloudDialogflowV2BatchUpdateEntitiesRequest', request_type_name='DialogflowProjectsAgentEntityTypesEntitiesBatchUpdateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)