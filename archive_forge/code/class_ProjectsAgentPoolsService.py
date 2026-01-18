from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storagetransfer.v1 import storagetransfer_v1_messages as messages
class ProjectsAgentPoolsService(base_api.BaseApiService):
    """Service class for the projects_agentPools resource."""
    _NAME = 'projects_agentPools'

    def __init__(self, client):
        super(StoragetransferV1.ProjectsAgentPoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an agent pool resource.

      Args:
        request: (StoragetransferProjectsAgentPoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AgentPool) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/agentPools', http_method='POST', method_id='storagetransfer.projects.agentPools.create', ordered_params=['projectId'], path_params=['projectId'], query_params=['agentPoolId'], relative_path='v1/projects/{+projectId}/agentPools', request_field='agentPool', request_type_name='StoragetransferProjectsAgentPoolsCreateRequest', response_type_name='AgentPool', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an agent pool.

      Args:
        request: (StoragetransferProjectsAgentPoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/agentPools/{agentPoolsId}', http_method='DELETE', method_id='storagetransfer.projects.agentPools.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='StoragetransferProjectsAgentPoolsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an agent pool.

      Args:
        request: (StoragetransferProjectsAgentPoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AgentPool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/agentPools/{agentPoolsId}', http_method='GET', method_id='storagetransfer.projects.agentPools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='StoragetransferProjectsAgentPoolsGetRequest', response_type_name='AgentPool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists agent pools.

      Args:
        request: (StoragetransferProjectsAgentPoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAgentPoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/agentPools', http_method='GET', method_id='storagetransfer.projects.agentPools.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/projects/{+projectId}/agentPools', request_field='', request_type_name='StoragetransferProjectsAgentPoolsListRequest', response_type_name='ListAgentPoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing agent pool resource.

      Args:
        request: (StoragetransferProjectsAgentPoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AgentPool) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/agentPools/{agentPoolsId}', http_method='PATCH', method_id='storagetransfer.projects.agentPools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='agentPool', request_type_name='StoragetransferProjectsAgentPoolsPatchRequest', response_type_name='AgentPool', supports_download=False)