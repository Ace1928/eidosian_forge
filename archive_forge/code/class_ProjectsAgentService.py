from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsAgentService(base_api.BaseApiService):
    """Service class for the projects_agent resource."""
    _NAME = 'projects_agent'

    def __init__(self, client):
        super(DialogflowV2.ProjectsAgentService, self).__init__(client)
        self._upload_configs = {}

    def Export(self, request, global_params=None):
        """Exports the specified agent to a ZIP file. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: ExportAgentResponse.

      Args:
        request: (DialogflowProjectsAgentExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent:export', http_method='POST', method_id='dialogflow.projects.agent.export', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/agent:export', request_field='googleCloudDialogflowV2ExportAgentRequest', request_type_name='DialogflowProjectsAgentExportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def GetFulfillment(self, request, global_params=None):
        """Retrieves the fulfillment.

      Args:
        request: (DialogflowProjectsAgentGetFulfillmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Fulfillment) The response message.
      """
        config = self.GetMethodConfig('GetFulfillment')
        return self._RunMethod(config, request, global_params=global_params)
    GetFulfillment.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/fulfillment', http_method='GET', method_id='dialogflow.projects.agent.getFulfillment', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsAgentGetFulfillmentRequest', response_type_name='GoogleCloudDialogflowV2Fulfillment', supports_download=False)

    def GetValidationResult(self, request, global_params=None):
        """Gets agent validation result. Agent validation is performed during training time and is updated automatically when training is completed.

      Args:
        request: (DialogflowProjectsAgentGetValidationResultRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ValidationResult) The response message.
      """
        config = self.GetMethodConfig('GetValidationResult')
        return self._RunMethod(config, request, global_params=global_params)
    GetValidationResult.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/validationResult', http_method='GET', method_id='dialogflow.projects.agent.getValidationResult', ordered_params=['parent'], path_params=['parent'], query_params=['languageCode'], relative_path='v2/{+parent}/agent/validationResult', request_field='', request_type_name='DialogflowProjectsAgentGetValidationResultRequest', response_type_name='GoogleCloudDialogflowV2ValidationResult', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports the specified agent from a ZIP file. Uploads new intents and entity types without deleting the existing ones. Intents and entity types with the same name are replaced with the new versions from ImportAgentRequest. After the import, the imported draft agent will be trained automatically (unless disabled in agent settings). However, once the import is done, training may not be completed yet. Please call TrainAgent and wait for the operation it returns in order to train explicitly. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) The operation only tracks when importing is complete, not when it is done training. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsAgentImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent:import', http_method='POST', method_id='dialogflow.projects.agent.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/agent:import', request_field='googleCloudDialogflowV2ImportAgentRequest', request_type_name='DialogflowProjectsAgentImportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Restore(self, request, global_params=None):
        """Restores the specified agent from a ZIP file. Replaces the current agent version with a new one. All the intents and entity types in the older version are deleted. After the restore, the restored draft agent will be trained automatically (unless disabled in agent settings). However, once the restore is done, training may not be completed yet. Please call TrainAgent and wait for the operation it returns in order to train explicitly. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) The operation only tracks when restoring is complete, not when it is done training. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsAgentRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Restore')
        return self._RunMethod(config, request, global_params=global_params)
    Restore.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent:restore', http_method='POST', method_id='dialogflow.projects.agent.restore', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/agent:restore', request_field='googleCloudDialogflowV2RestoreAgentRequest', request_type_name='DialogflowProjectsAgentRestoreRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Search(self, request, global_params=None):
        """Returns the list of agents. Since there is at most one conversational agent per project, this method is useful primarily for listing all agents across projects the caller has access to. One can achieve that with a wildcard project collection id "-". Refer to [List Sub-Collections](https://cloud.google.com/apis/design/design_patterns#list_sub-collections).

      Args:
        request: (DialogflowProjectsAgentSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SearchAgentsResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent:search', http_method='GET', method_id='dialogflow.projects.agent.search', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/agent:search', request_field='', request_type_name='DialogflowProjectsAgentSearchRequest', response_type_name='GoogleCloudDialogflowV2SearchAgentsResponse', supports_download=False)

    def Train(self, request, global_params=None):
        """Trains the specified agent. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsAgentTrainRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Train')
        return self._RunMethod(config, request, global_params=global_params)
    Train.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent:train', http_method='POST', method_id='dialogflow.projects.agent.train', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/agent:train', request_field='googleCloudDialogflowV2TrainAgentRequest', request_type_name='DialogflowProjectsAgentTrainRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def UpdateFulfillment(self, request, global_params=None):
        """Updates the fulfillment.

      Args:
        request: (DialogflowProjectsAgentUpdateFulfillmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Fulfillment) The response message.
      """
        config = self.GetMethodConfig('UpdateFulfillment')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateFulfillment.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/fulfillment', http_method='PATCH', method_id='dialogflow.projects.agent.updateFulfillment', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Fulfillment', request_type_name='DialogflowProjectsAgentUpdateFulfillmentRequest', response_type_name='GoogleCloudDialogflowV2Fulfillment', supports_download=False)