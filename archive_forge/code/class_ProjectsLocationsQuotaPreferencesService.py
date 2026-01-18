from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudquotas.v1 import cloudquotas_v1_messages as messages
class ProjectsLocationsQuotaPreferencesService(base_api.BaseApiService):
    """Service class for the projects_locations_quotaPreferences resource."""
    _NAME = 'projects_locations_quotaPreferences'

    def __init__(self, client):
        super(CloudquotasV1.ProjectsLocationsQuotaPreferencesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new QuotaPreference that declares the desired value for a quota.

      Args:
        request: (CloudquotasProjectsLocationsQuotaPreferencesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QuotaPreference) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/quotaPreferences', http_method='POST', method_id='cloudquotas.projects.locations.quotaPreferences.create', ordered_params=['parent'], path_params=['parent'], query_params=['ignoreSafetyChecks', 'quotaPreferenceId'], relative_path='v1/{+parent}/quotaPreferences', request_field='quotaPreference', request_type_name='CloudquotasProjectsLocationsQuotaPreferencesCreateRequest', response_type_name='QuotaPreference', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single QuotaPreference.

      Args:
        request: (CloudquotasProjectsLocationsQuotaPreferencesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QuotaPreference) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/quotaPreferences/{quotaPreferencesId}', http_method='GET', method_id='cloudquotas.projects.locations.quotaPreferences.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudquotasProjectsLocationsQuotaPreferencesGetRequest', response_type_name='QuotaPreference', supports_download=False)

    def List(self, request, global_params=None):
        """Lists QuotaPreferences in a given project, folder or organization.

      Args:
        request: (CloudquotasProjectsLocationsQuotaPreferencesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListQuotaPreferencesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/quotaPreferences', http_method='GET', method_id='cloudquotas.projects.locations.quotaPreferences.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/quotaPreferences', request_field='', request_type_name='CloudquotasProjectsLocationsQuotaPreferencesListRequest', response_type_name='ListQuotaPreferencesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single QuotaPreference. It can updates the config in any states, not just the ones pending approval.

      Args:
        request: (CloudquotasProjectsLocationsQuotaPreferencesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QuotaPreference) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/quotaPreferences/{quotaPreferencesId}', http_method='PATCH', method_id='cloudquotas.projects.locations.quotaPreferences.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'ignoreSafetyChecks', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='quotaPreference', request_type_name='CloudquotasProjectsLocationsQuotaPreferencesPatchRequest', response_type_name='QuotaPreference', supports_download=False)