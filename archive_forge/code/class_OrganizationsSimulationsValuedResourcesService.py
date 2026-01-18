from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class OrganizationsSimulationsValuedResourcesService(base_api.BaseApiService):
    """Service class for the organizations_simulations_valuedResources resource."""
    _NAME = 'organizations_simulations_valuedResources'

    def __init__(self, client):
        super(SecuritycenterV2.OrganizationsSimulationsValuedResourcesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get the valued resource by name.

      Args:
        request: (SecuritycenterOrganizationsSimulationsValuedResourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValuedResource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/simulations/{simulationsId}/valuedResources/{valuedResourcesId}', http_method='GET', method_id='securitycenter.organizations.simulations.valuedResources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsSimulationsValuedResourcesGetRequest', response_type_name='ValuedResource', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the valued resources for a set of simulation results and filter.

      Args:
        request: (SecuritycenterOrganizationsSimulationsValuedResourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListValuedResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/simulations/{simulationsId}/valuedResources', http_method='GET', method_id='securitycenter.organizations.simulations.valuedResources.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/valuedResources', request_field='', request_type_name='SecuritycenterOrganizationsSimulationsValuedResourcesListRequest', response_type_name='ListValuedResourcesResponse', supports_download=False)