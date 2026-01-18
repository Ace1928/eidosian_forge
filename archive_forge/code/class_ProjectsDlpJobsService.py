from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsDlpJobsService(base_api.BaseApiService):
    """Service class for the projects_dlpJobs resource."""
    _NAME = 'projects_dlpJobs'

    def __init__(self, client):
        super(DlpV2.ProjectsDlpJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Starts asynchronous cancellation on a long-running DlpJob. The server makes a best effort to cancel the DlpJob, but success is not guaranteed. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more.

      Args:
        request: (DlpProjectsDlpJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/dlpJobs/{dlpJobsId}:cancel', http_method='POST', method_id='dlp.projects.dlpJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:cancel', request_field='googlePrivacyDlpV2CancelDlpJobRequest', request_type_name='DlpProjectsDlpJobsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new job to inspect storage or calculate risk metrics. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more. When no InfoTypes or CustomInfoTypes are specified in inspect jobs, the system will automatically choose what detectors to run. By default this may be all types, but may change over time as detectors are updated.

      Args:
        request: (DlpProjectsDlpJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DlpJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/dlpJobs', http_method='POST', method_id='dlp.projects.dlpJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/dlpJobs', request_field='googlePrivacyDlpV2CreateDlpJobRequest', request_type_name='DlpProjectsDlpJobsCreateRequest', response_type_name='GooglePrivacyDlpV2DlpJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a long-running DlpJob. This method indicates that the client is no longer interested in the DlpJob result. The job will be canceled if possible. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more.

      Args:
        request: (DlpProjectsDlpJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/dlpJobs/{dlpJobsId}', http_method='DELETE', method_id='dlp.projects.dlpJobs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsDlpJobsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running DlpJob. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more.

      Args:
        request: (DlpProjectsDlpJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DlpJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/dlpJobs/{dlpJobsId}', http_method='GET', method_id='dlp.projects.dlpJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsDlpJobsGetRequest', response_type_name='GooglePrivacyDlpV2DlpJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DlpJobs that match the specified filter in the request. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more.

      Args:
        request: (DlpProjectsDlpJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListDlpJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/dlpJobs', http_method='GET', method_id='dlp.projects.dlpJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'locationId', 'orderBy', 'pageSize', 'pageToken', 'type'], relative_path='v2/{+parent}/dlpJobs', request_field='', request_type_name='DlpProjectsDlpJobsListRequest', response_type_name='GooglePrivacyDlpV2ListDlpJobsResponse', supports_download=False)