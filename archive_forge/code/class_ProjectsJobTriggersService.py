from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsJobTriggersService(base_api.BaseApiService):
    """Service class for the projects_jobTriggers resource."""
    _NAME = 'projects_jobTriggers'

    def __init__(self, client):
        super(DlpV2.ProjectsJobTriggersService, self).__init__(client)
        self._upload_configs = {}

    def Activate(self, request, global_params=None):
        """Activate a job trigger. Causes the immediate execute of a trigger instead of waiting on the trigger event to occur.

      Args:
        request: (DlpProjectsJobTriggersActivateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DlpJob) The response message.
      """
        config = self.GetMethodConfig('Activate')
        return self._RunMethod(config, request, global_params=global_params)
    Activate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/jobTriggers/{jobTriggersId}:activate', http_method='POST', method_id='dlp.projects.jobTriggers.activate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:activate', request_field='googlePrivacyDlpV2ActivateJobTriggerRequest', request_type_name='DlpProjectsJobTriggersActivateRequest', response_type_name='GooglePrivacyDlpV2DlpJob', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a job trigger to run DLP actions such as scanning storage for sensitive information on a set schedule. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpProjectsJobTriggersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2JobTrigger) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/jobTriggers', http_method='POST', method_id='dlp.projects.jobTriggers.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/jobTriggers', request_field='googlePrivacyDlpV2CreateJobTriggerRequest', request_type_name='DlpProjectsJobTriggersCreateRequest', response_type_name='GooglePrivacyDlpV2JobTrigger', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a job trigger. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpProjectsJobTriggersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/jobTriggers/{jobTriggersId}', http_method='DELETE', method_id='dlp.projects.jobTriggers.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsJobTriggersDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a job trigger. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpProjectsJobTriggersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2JobTrigger) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/jobTriggers/{jobTriggersId}', http_method='GET', method_id='dlp.projects.jobTriggers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsJobTriggersGetRequest', response_type_name='GooglePrivacyDlpV2JobTrigger', supports_download=False)

    def List(self, request, global_params=None):
        """Lists job triggers. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpProjectsJobTriggersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListJobTriggersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/jobTriggers', http_method='GET', method_id='dlp.projects.jobTriggers.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'locationId', 'orderBy', 'pageSize', 'pageToken', 'type'], relative_path='v2/{+parent}/jobTriggers', request_field='', request_type_name='DlpProjectsJobTriggersListRequest', response_type_name='GooglePrivacyDlpV2ListJobTriggersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a job trigger. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpProjectsJobTriggersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2JobTrigger) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/jobTriggers/{jobTriggersId}', http_method='PATCH', method_id='dlp.projects.jobTriggers.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='googlePrivacyDlpV2UpdateJobTriggerRequest', request_type_name='DlpProjectsJobTriggersPatchRequest', response_type_name='GooglePrivacyDlpV2JobTrigger', supports_download=False)