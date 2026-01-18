from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v2alpha import serviceusage_v2alpha_messages as messages
class ServicesGroupsService(base_api.BaseApiService):
    """Service class for the services_groups resource."""
    _NAME = 'services_groups'

    def __init__(self, client):
        super(ServiceusageV2alpha.ServicesGroupsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List service groups owned by the given service.

      Args:
        request: (ServiceusageServicesGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/{v2alphaId}/{v2alphaId1}/services/{servicesId}/groups', http_method='GET', method_id='serviceusage.services.groups.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v2alpha/{+parent}/groups', request_field='', request_type_name='ServiceusageServicesGroupsListRequest', response_type_name='ListServiceGroupsResponse', supports_download=False)