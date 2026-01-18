from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class OrganizationsSimulationsService(base_api.BaseApiService):
    """Service class for the organizations_simulations resource."""
    _NAME = 'organizations_simulations'

    def __init__(self, client):
        super(SecuritycenterV2.OrganizationsSimulationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get the simulation by name or the latest simulation for the given organization.

      Args:
        request: (SecuritycenterOrganizationsSimulationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Simulation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/simulations/{simulationsId}', http_method='GET', method_id='securitycenter.organizations.simulations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsSimulationsGetRequest', response_type_name='Simulation', supports_download=False)