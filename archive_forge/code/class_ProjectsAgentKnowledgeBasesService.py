from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsAgentKnowledgeBasesService(base_api.BaseApiService):
    """Service class for the projects_agent_knowledgeBases resource."""
    _NAME = 'projects_agent_knowledgeBases'

    def __init__(self, client):
        super(DialogflowV2.ProjectsAgentKnowledgeBasesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a knowledge base.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2KnowledgeBase) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases', http_method='POST', method_id='dialogflow.projects.agent.knowledgeBases.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/knowledgeBases', request_field='googleCloudDialogflowV2KnowledgeBase', request_type_name='DialogflowProjectsAgentKnowledgeBasesCreateRequest', response_type_name='GoogleCloudDialogflowV2KnowledgeBase', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified knowledge base.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases/{knowledgeBasesId}', http_method='DELETE', method_id='dialogflow.projects.agent.knowledgeBases.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsAgentKnowledgeBasesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified knowledge base.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2KnowledgeBase) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases/{knowledgeBasesId}', http_method='GET', method_id='dialogflow.projects.agent.knowledgeBases.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsAgentKnowledgeBasesGetRequest', response_type_name='GoogleCloudDialogflowV2KnowledgeBase', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all knowledge bases of the specified agent.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListKnowledgeBasesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases', http_method='GET', method_id='dialogflow.projects.agent.knowledgeBases.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/knowledgeBases', request_field='', request_type_name='DialogflowProjectsAgentKnowledgeBasesListRequest', response_type_name='GoogleCloudDialogflowV2ListKnowledgeBasesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified knowledge base.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2KnowledgeBase) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases/{knowledgeBasesId}', http_method='PATCH', method_id='dialogflow.projects.agent.knowledgeBases.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2KnowledgeBase', request_type_name='DialogflowProjectsAgentKnowledgeBasesPatchRequest', response_type_name='GoogleCloudDialogflowV2KnowledgeBase', supports_download=False)