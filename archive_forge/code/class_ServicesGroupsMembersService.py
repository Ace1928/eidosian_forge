from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v2alpha import serviceusage_v2alpha_messages as messages
class ServicesGroupsMembersService(base_api.BaseApiService):
    """Service class for the services_groups_members resource."""
    _NAME = 'services_groups_members'

    def __init__(self, client):
        super(ServiceusageV2alpha.ServicesGroupsMembersService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List members for the given service group. The service group is a producer defined service group.

      Args:
        request: (ServiceusageServicesGroupsMembersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGroupMembersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/{v2alphaId}/{v2alphaId1}/services/{servicesId}/groups/{groupsId}/members', http_method='GET', method_id='serviceusage.services.groups.members.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v2alpha/{+parent}/members', request_field='', request_type_name='ServiceusageServicesGroupsMembersListRequest', response_type_name='ListGroupMembersResponse', supports_download=False)