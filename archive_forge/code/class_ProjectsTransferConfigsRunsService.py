from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigquerydatatransfer.v1 import bigquerydatatransfer_v1_messages as messages
class ProjectsTransferConfigsRunsService(base_api.BaseApiService):
    """Service class for the projects_transferConfigs_runs resource."""
    _NAME = 'projects_transferConfigs_runs'

    def __init__(self, client):
        super(BigquerydatatransferV1.ProjectsTransferConfigsRunsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified transfer run.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsRunsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs/{transferConfigsId}/runs/{runsId}', http_method='DELETE', method_id='bigquerydatatransfer.projects.transferConfigs.runs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BigquerydatatransferProjectsTransferConfigsRunsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns information about the particular transfer run.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TransferRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs/{transferConfigsId}/runs/{runsId}', http_method='GET', method_id='bigquerydatatransfer.projects.transferConfigs.runs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BigquerydatatransferProjectsTransferConfigsRunsGetRequest', response_type_name='TransferRun', supports_download=False)

    def List(self, request, global_params=None):
        """Returns information about running and completed transfer runs.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTransferRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs/{transferConfigsId}/runs', http_method='GET', method_id='bigquerydatatransfer.projects.transferConfigs.runs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'runAttempt', 'states'], relative_path='v1/{+parent}/runs', request_field='', request_type_name='BigquerydatatransferProjectsTransferConfigsRunsListRequest', response_type_name='ListTransferRunsResponse', supports_download=False)