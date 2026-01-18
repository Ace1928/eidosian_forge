from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsAgentEnvironmentsUsersSessionsContextsService(base_api.BaseApiService):
    """Service class for the projects_locations_agent_environments_users_sessions_contexts resource."""
    _NAME = 'projects_locations_agent_environments_users_sessions_contexts'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsAgentEnvironmentsUsersSessionsContextsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a context. If the specified context already exists, overrides the context.

      Args:
        request: (DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Context) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/contexts', http_method='POST', method_id='dialogflow.projects.locations.agent.environments.users.sessions.contexts.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/contexts', request_field='googleCloudDialogflowV2Context', request_type_name='DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsCreateRequest', response_type_name='GoogleCloudDialogflowV2Context', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified context.

      Args:
        request: (DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/contexts/{contextsId}', http_method='DELETE', method_id='dialogflow.projects.locations.agent.environments.users.sessions.contexts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified context.

      Args:
        request: (DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Context) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/contexts/{contextsId}', http_method='GET', method_id='dialogflow.projects.locations.agent.environments.users.sessions.contexts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsGetRequest', response_type_name='GoogleCloudDialogflowV2Context', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all contexts in the specified session.

      Args:
        request: (DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListContextsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/contexts', http_method='GET', method_id='dialogflow.projects.locations.agent.environments.users.sessions.contexts.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/contexts', request_field='', request_type_name='DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsListRequest', response_type_name='GoogleCloudDialogflowV2ListContextsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified context.

      Args:
        request: (DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Context) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/contexts/{contextsId}', http_method='PATCH', method_id='dialogflow.projects.locations.agent.environments.users.sessions.contexts.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Context', request_type_name='DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsContextsPatchRequest', response_type_name='GoogleCloudDialogflowV2Context', supports_download=False)