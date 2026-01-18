from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsAgentEnvironmentsUsersSessionsEntityTypesService(base_api.BaseApiService):
    """Service class for the projects_agent_environments_users_sessions_entityTypes resource."""
    _NAME = 'projects_agent_environments_users_sessions_entityTypes'

    def __init__(self, client):
        super(DialogflowV2.ProjectsAgentEnvironmentsUsersSessionsEntityTypesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a session entity type. If the specified session entity type already exists, overrides the session entity type. This method doesn't work with Google Assistant integration. Contact Dialogflow support if you need to use session entities with Google Assistant integration.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SessionEntityType) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/entityTypes', http_method='POST', method_id='dialogflow.projects.agent.environments.users.sessions.entityTypes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/entityTypes', request_field='googleCloudDialogflowV2SessionEntityType', request_type_name='DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesCreateRequest', response_type_name='GoogleCloudDialogflowV2SessionEntityType', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified session entity type. This method doesn't work with Google Assistant integration. Contact Dialogflow support if you need to use session entities with Google Assistant integration.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/entityTypes/{entityTypesId}', http_method='DELETE', method_id='dialogflow.projects.agent.environments.users.sessions.entityTypes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified session entity type. This method doesn't work with Google Assistant integration. Contact Dialogflow support if you need to use session entities with Google Assistant integration.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SessionEntityType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/entityTypes/{entityTypesId}', http_method='GET', method_id='dialogflow.projects.agent.environments.users.sessions.entityTypes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesGetRequest', response_type_name='GoogleCloudDialogflowV2SessionEntityType', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all session entity types in the specified session. This method doesn't work with Google Assistant integration. Contact Dialogflow support if you need to use session entities with Google Assistant integration.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListSessionEntityTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/entityTypes', http_method='GET', method_id='dialogflow.projects.agent.environments.users.sessions.entityTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/entityTypes', request_field='', request_type_name='DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesListRequest', response_type_name='GoogleCloudDialogflowV2ListSessionEntityTypesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified session entity type. This method doesn't work with Google Assistant integration. Contact Dialogflow support if you need to use session entities with Google Assistant integration.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SessionEntityType) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/entityTypes/{entityTypesId}', http_method='PATCH', method_id='dialogflow.projects.agent.environments.users.sessions.entityTypes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2SessionEntityType', request_type_name='DialogflowProjectsAgentEnvironmentsUsersSessionsEntityTypesPatchRequest', response_type_name='GoogleCloudDialogflowV2SessionEntityType', supports_download=False)