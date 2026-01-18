from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class OrganizationsSimulationsAttackExposureResultsValuedResourcesService(base_api.BaseApiService):
    """Service class for the organizations_simulations_attackExposureResults_valuedResources resource."""
    _NAME = 'organizations_simulations_attackExposureResults_valuedResources'

    def __init__(self, client):
        super(SecuritycenterV2.OrganizationsSimulationsAttackExposureResultsValuedResourcesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the valued resources for a set of simulation results and filter.

      Args:
        request: (SecuritycenterOrganizationsSimulationsAttackExposureResultsValuedResourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListValuedResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/simulations/{simulationsId}/attackExposureResults/{attackExposureResultsId}/valuedResources', http_method='GET', method_id='securitycenter.organizations.simulations.attackExposureResults.valuedResources.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/valuedResources', request_field='', request_type_name='SecuritycenterOrganizationsSimulationsAttackExposureResultsValuedResourcesListRequest', response_type_name='ListValuedResourcesResponse', supports_download=False)