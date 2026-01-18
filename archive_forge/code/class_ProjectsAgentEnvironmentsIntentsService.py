from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsAgentEnvironmentsIntentsService(base_api.BaseApiService):
    """Service class for the projects_agent_environments_intents resource."""
    _NAME = 'projects_agent_environments_intents'

    def __init__(self, client):
        super(DialogflowV2.ProjectsAgentEnvironmentsIntentsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns the list of all intents in the specified agent.

      Args:
        request: (DialogflowProjectsAgentEnvironmentsIntentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListIntentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/environments/{environmentsId}/intents', http_method='GET', method_id='dialogflow.projects.agent.environments.intents.list', ordered_params=['parent'], path_params=['parent'], query_params=['intentView', 'languageCode', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/intents', request_field='', request_type_name='DialogflowProjectsAgentEnvironmentsIntentsListRequest', response_type_name='GoogleCloudDialogflowV2ListIntentsResponse', supports_download=False)