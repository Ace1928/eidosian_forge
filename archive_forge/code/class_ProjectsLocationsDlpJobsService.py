from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsLocationsDlpJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_dlpJobs resource."""
    _NAME = 'projects_locations_dlpJobs'

    def __init__(self, client):
        super(DlpV2.ProjectsLocationsDlpJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Starts asynchronous cancellation on a long-running DlpJob. The server makes a best effort to cancel the DlpJob, but success is not guaranteed. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more.

      Args:
        request: (DlpProjectsLocationsDlpJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/dlpJobs/{dlpJobsId}:cancel', http_method='POST', method_id='dlp.projects.locations.dlpJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:cancel', request_field='googlePrivacyDlpV2CancelDlpJobRequest', request_type_name='DlpProjectsLocationsDlpJobsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new job to inspect storage or calculate risk metrics. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more. When no InfoTypes or CustomInfoTypes are specified in inspect jobs, the system will automatically choose what detectors to run. By default this may be all types, but may change over time as detectors are updated.

      Args:
        request: (DlpProjectsLocationsDlpJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DlpJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/dlpJobs', http_method='POST', method_id='dlp.projects.locations.dlpJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/dlpJobs', request_field='googlePrivacyDlpV2CreateDlpJobRequest', request_type_name='DlpProjectsLocationsDlpJobsCreateRequest', response_type_name='GooglePrivacyDlpV2DlpJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a long-running DlpJob. This method indicates that the client is no longer interested in the DlpJob result. The job will be canceled if possible. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more.

      Args:
        request: (DlpProjectsLocationsDlpJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/dlpJobs/{dlpJobsId}', http_method='DELETE', method_id='dlp.projects.locations.dlpJobs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsDlpJobsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Finish(self, request, global_params=None):
        """Finish a running hybrid DlpJob. Triggers the finalization steps and running of any enabled actions that have not yet run.

      Args:
        request: (DlpProjectsLocationsDlpJobsFinishRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Finish')
        return self._RunMethod(config, request, global_params=global_params)
    Finish.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/dlpJobs/{dlpJobsId}:finish', http_method='POST', method_id='dlp.projects.locations.dlpJobs.finish', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:finish', request_field='googlePrivacyDlpV2FinishDlpJobRequest', request_type_name='DlpProjectsLocationsDlpJobsFinishRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running DlpJob. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more.

      Args:
        request: (DlpProjectsLocationsDlpJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DlpJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/dlpJobs/{dlpJobsId}', http_method='GET', method_id='dlp.projects.locations.dlpJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsDlpJobsGetRequest', response_type_name='GooglePrivacyDlpV2DlpJob', supports_download=False)

    def HybridInspect(self, request, global_params=None):
        """Inspect hybrid content and store findings to a job. To review the findings, inspect the job. Inspection will occur asynchronously.

      Args:
        request: (DlpProjectsLocationsDlpJobsHybridInspectRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2HybridInspectResponse) The response message.
      """
        config = self.GetMethodConfig('HybridInspect')
        return self._RunMethod(config, request, global_params=global_params)
    HybridInspect.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/dlpJobs/{dlpJobsId}:hybridInspect', http_method='POST', method_id='dlp.projects.locations.dlpJobs.hybridInspect', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:hybridInspect', request_field='googlePrivacyDlpV2HybridInspectDlpJobRequest', request_type_name='DlpProjectsLocationsDlpJobsHybridInspectRequest', response_type_name='GooglePrivacyDlpV2HybridInspectResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DlpJobs that match the specified filter in the request. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more.

      Args:
        request: (DlpProjectsLocationsDlpJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListDlpJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/dlpJobs', http_method='GET', method_id='dlp.projects.locations.dlpJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'locationId', 'orderBy', 'pageSize', 'pageToken', 'type'], relative_path='v2/{+parent}/dlpJobs', request_field='', request_type_name='DlpProjectsLocationsDlpJobsListRequest', response_type_name='GooglePrivacyDlpV2ListDlpJobsResponse', supports_download=False)