from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class OrganizationsLocationsJobTriggersService(base_api.BaseApiService):
    """Service class for the organizations_locations_jobTriggers resource."""
    _NAME = 'organizations_locations_jobTriggers'

    def __init__(self, client):
        super(DlpV2.OrganizationsLocationsJobTriggersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a job trigger to run DLP actions such as scanning storage for sensitive information on a set schedule. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpOrganizationsLocationsJobTriggersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2JobTrigger) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/jobTriggers', http_method='POST', method_id='dlp.organizations.locations.jobTriggers.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/jobTriggers', request_field='googlePrivacyDlpV2CreateJobTriggerRequest', request_type_name='DlpOrganizationsLocationsJobTriggersCreateRequest', response_type_name='GooglePrivacyDlpV2JobTrigger', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a job trigger. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpOrganizationsLocationsJobTriggersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/jobTriggers/{jobTriggersId}', http_method='DELETE', method_id='dlp.organizations.locations.jobTriggers.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpOrganizationsLocationsJobTriggersDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a job trigger. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpOrganizationsLocationsJobTriggersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2JobTrigger) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/jobTriggers/{jobTriggersId}', http_method='GET', method_id='dlp.organizations.locations.jobTriggers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpOrganizationsLocationsJobTriggersGetRequest', response_type_name='GooglePrivacyDlpV2JobTrigger', supports_download=False)

    def List(self, request, global_params=None):
        """Lists job triggers. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpOrganizationsLocationsJobTriggersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListJobTriggersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/jobTriggers', http_method='GET', method_id='dlp.organizations.locations.jobTriggers.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'locationId', 'orderBy', 'pageSize', 'pageToken', 'type'], relative_path='v2/{+parent}/jobTriggers', request_field='', request_type_name='DlpOrganizationsLocationsJobTriggersListRequest', response_type_name='GooglePrivacyDlpV2ListJobTriggersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a job trigger. See https://cloud.google.com/sensitive-data-protection/docs/creating-job-triggers to learn more.

      Args:
        request: (DlpOrganizationsLocationsJobTriggersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2JobTrigger) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/jobTriggers/{jobTriggersId}', http_method='PATCH', method_id='dlp.organizations.locations.jobTriggers.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='googlePrivacyDlpV2UpdateJobTriggerRequest', request_type_name='DlpOrganizationsLocationsJobTriggersPatchRequest', response_type_name='GooglePrivacyDlpV2JobTrigger', supports_download=False)