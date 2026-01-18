from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.marketplacesolutions.v1alpha1 import marketplacesolutions_v1alpha1_messages as messages
class ProjectsLocationsPowerSshKeysService(base_api.BaseApiService):
    """Service class for the projects_locations_powerSshKeys resource."""
    _NAME = 'projects_locations_powerSshKeys'

    def __init__(self, client):
        super(MarketplacesolutionsV1alpha1.ProjectsLocationsPowerSshKeysService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get details about a single Power SSH Key.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerSshKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PowerSSHKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerSshKeys/{powerSshKeysId}', http_method='GET', method_id='marketplacesolutions.projects.locations.powerSshKeys.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerSshKeysGetRequest', response_type_name='PowerSSHKey', supports_download=False)

    def List(self, request, global_params=None):
        """List SSH Keys in a given project and location from Power.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerSshKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPowerSSHKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/powerSshKeys', http_method='GET', method_id='marketplacesolutions.projects.locations.powerSshKeys.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/powerSshKeys', request_field='', request_type_name='MarketplacesolutionsProjectsLocationsPowerSshKeysListRequest', response_type_name='ListPowerSSHKeysResponse', supports_download=False)