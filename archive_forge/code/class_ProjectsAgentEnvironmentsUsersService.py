from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsAgentEnvironmentsUsersService(base_api.BaseApiService):
    """Service class for the projects_agent_environments_users resource."""
    _NAME = 'projects_agent_environments_users'

    def __init__(self, client):
        super(DialogflowV2.ProjectsAgentEnvironmentsUsersService, self).__init__(client)
        self._upload_configs = {}