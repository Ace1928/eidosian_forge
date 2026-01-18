from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsAgentEnvironmentsService(base_api.BaseApiService):
    """Service class for the projects_agent_environments resource."""
    _NAME = 'projects_agent_environments'

    def __init__(self, client):
        super(DialogflowV2.ProjectsAgentEnvironmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an agent environment.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Environment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments', http_method='POST', method_id='dialogflow.projects.agent.environments.create', ordered_params=['parent'], path_params=['parent'], query_params=['environmentId'], relative_path='v2/{+parent}/environments', request_field='googleCloudDialogflowV2Environment', request_type_name='DialogflowProjectsAgentEnvironmentsCreateRequest', response_type_name='GoogleCloudDialogflowV2Environment', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified agent environment.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}', http_method='DELETE', method_id='dialogflow.projects.agent.environments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsAgentEnvironmentsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified agent environment.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Environment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}', http_method='GET', method_id='dialogflow.projects.agent.environments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsAgentEnvironmentsGetRequest', response_type_name='GoogleCloudDialogflowV2Environment', supports_download=False)

    def GetHistory(self, request, global_params=None):
        """Gets the history of the specified environment.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsGetHistoryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2EnvironmentHistory) The response message.
      """
        config = self.GetMethodConfig('GetHistory')
        return self._RunMethod(config, request, global_params=global_params)
    GetHistory.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}/history', http_method='GET', method_id='dialogflow.projects.agent.environments.getHistory', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/history', request_field='', request_type_name='DialogflowProjectsAgentEnvironmentsGetHistoryRequest', response_type_name='GoogleCloudDialogflowV2EnvironmentHistory', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all non-default environments of the specified agent.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListEnvironmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments', http_method='GET', method_id='dialogflow.projects.agent.environments.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/environments', request_field='', request_type_name='DialogflowProjectsAgentEnvironmentsListRequest', response_type_name='GoogleCloudDialogflowV2ListEnvironmentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified agent environment. This method allows you to deploy new agent versions into the environment. When an environment is pointed to a new agent version by setting `environment.agent_version`, the environment is temporarily set to the `LOADING` state. During that time, the environment continues serving the previous version of the agent. After the new agent version is done loading, the environment is set back to the `RUNNING` state. You can use "-" as Environment ID in environment name to update an agent version in the default environment. WARNING: this will negate all recent changes to the draft agent and can't be undone. You may want to save the draft agent to a version before calling this method.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Environment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}', http_method='PATCH', method_id='dialogflow.projects.agent.environments.patch', ordered_params=['name'], path_params=['name'], query_params=['allowLoadToDraftAndDiscardChanges', 'updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Environment', request_type_name='DialogflowProjectsAgentEnvironmentsPatchRequest', response_type_name='GoogleCloudDialogflowV2Environment', supports_download=False)