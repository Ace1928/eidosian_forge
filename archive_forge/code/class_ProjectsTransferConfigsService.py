from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigquerydatatransfer.v1 import bigquerydatatransfer_v1_messages as messages
class ProjectsTransferConfigsService(base_api.BaseApiService):
    """Service class for the projects_transferConfigs resource."""
    _NAME = 'projects_transferConfigs'

    def __init__(self, client):
        super(BigquerydatatransferV1.ProjectsTransferConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new data transfer configuration.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TransferConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs', http_method='POST', method_id='bigquerydatatransfer.projects.transferConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['authorizationCode', 'serviceAccountName', 'versionInfo'], relative_path='v1/{+parent}/transferConfigs', request_field='transferConfig', request_type_name='BigquerydatatransferProjectsTransferConfigsCreateRequest', response_type_name='TransferConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a data transfer configuration, including any associated transfer runs and logs.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs/{transferConfigsId}', http_method='DELETE', method_id='bigquerydatatransfer.projects.transferConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BigquerydatatransferProjectsTransferConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns information about a data transfer config.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TransferConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs/{transferConfigsId}', http_method='GET', method_id='bigquerydatatransfer.projects.transferConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BigquerydatatransferProjectsTransferConfigsGetRequest', response_type_name='TransferConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Returns information about all transfer configs owned by a project in the specified location.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTransferConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs', http_method='GET', method_id='bigquerydatatransfer.projects.transferConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['dataSourceIds', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/transferConfigs', request_field='', request_type_name='BigquerydatatransferProjectsTransferConfigsListRequest', response_type_name='ListTransferConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a data transfer configuration. All fields must be set, even if they are not updated.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TransferConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs/{transferConfigsId}', http_method='PATCH', method_id='bigquerydatatransfer.projects.transferConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['authorizationCode', 'serviceAccountName', 'updateMask', 'versionInfo'], relative_path='v1/{+name}', request_field='transferConfig', request_type_name='BigquerydatatransferProjectsTransferConfigsPatchRequest', response_type_name='TransferConfig', supports_download=False)

    def ScheduleRuns(self, request, global_params=None):
        """Creates transfer runs for a time range [start_time, end_time]. For each date - or whatever granularity the data source supports - in the range, one transfer run is created. Note that runs are created per UTC time in the time range. DEPRECATED: use StartManualTransferRuns instead.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsScheduleRunsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ScheduleTransferRunsResponse) The response message.
      """
        config = self.GetMethodConfig('ScheduleRuns')
        return self._RunMethod(config, request, global_params=global_params)
    ScheduleRuns.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs/{transferConfigsId}:scheduleRuns', http_method='POST', method_id='bigquerydatatransfer.projects.transferConfigs.scheduleRuns', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:scheduleRuns', request_field='scheduleTransferRunsRequest', request_type_name='BigquerydatatransferProjectsTransferConfigsScheduleRunsRequest', response_type_name='ScheduleTransferRunsResponse', supports_download=False)

    def StartManualRuns(self, request, global_params=None):
        """Start manual transfer runs to be executed now with schedule_time equal to current time. The transfer runs can be created for a time range where the run_time is between start_time (inclusive) and end_time (exclusive), or for a specific run_time.

      Args:
        request: (BigquerydatatransferProjectsTransferConfigsStartManualRunsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StartManualTransferRunsResponse) The response message.
      """
        config = self.GetMethodConfig('StartManualRuns')
        return self._RunMethod(config, request, global_params=global_params)
    StartManualRuns.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/transferConfigs/{transferConfigsId}:startManualRuns', http_method='POST', method_id='bigquerydatatransfer.projects.transferConfigs.startManualRuns', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:startManualRuns', request_field='startManualTransferRunsRequest', request_type_name='BigquerydatatransferProjectsTransferConfigsStartManualRunsRequest', response_type_name='StartManualTransferRunsResponse', supports_download=False)