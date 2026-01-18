from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class ProjectsLocationsServiceaccountsService(base_api.BaseApiService):
    """Service class for the projects_locations_serviceaccounts resource."""
    _NAME = 'projects_locations_serviceaccounts'

    def __init__(self, client):
        super(AnthoseventsV1.ProjectsLocationsServiceaccountsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Rpc to retrieve service account.

      Args:
        request: (AnthoseventsProjectsLocationsServiceaccountsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccount) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceaccounts/{serviceaccountsId}', http_method='GET', method_id='anthosevents.projects.locations.serviceaccounts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AnthoseventsProjectsLocationsServiceaccountsGetRequest', response_type_name='ServiceAccount', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list Service Accounts.

      Args:
        request: (AnthoseventsProjectsLocationsServiceaccountsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceAccountsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceaccounts', http_method='GET', method_id='anthosevents.projects.locations.serviceaccounts.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/serviceaccounts', request_field='', request_type_name='AnthoseventsProjectsLocationsServiceaccountsListRequest', response_type_name='ListServiceAccountsResponse', supports_download=False)