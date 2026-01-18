from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v2alpha import serviceusage_v2alpha_messages as messages
class ServicesAncestorGroupsService(base_api.BaseApiService):
    """Service class for the services_ancestorGroups resource."""
    _NAME = 'services_ancestorGroups'

    def __init__(self, client):
        super(ServiceusageV2alpha.ServicesAncestorGroupsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List the ancestor groups that depend on the service. This lists the groups that include the parent service directly or which include a group for which the specified service is a descendant service.

      Args:
        request: (ServiceusageServicesAncestorGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAncestorGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/{v2alphaId}/{v2alphaId1}/services/{servicesId}/ancestorGroups', http_method='GET', method_id='serviceusage.services.ancestorGroups.list', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v2alpha/{+name}/ancestorGroups', request_field='', request_type_name='ServiceusageServicesAncestorGroupsListRequest', response_type_name='ListAncestorGroupsResponse', supports_download=False)