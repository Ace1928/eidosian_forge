from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class NetworkEndpointGroupsService(base_api.BaseApiService):
    """Service class for the networkEndpointGroups resource."""
    _NAME = 'networkEndpointGroups'

    def __init__(self, client):
        super(ComputeBeta.NetworkEndpointGroupsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of network endpoint groups and sorts them by zone. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeNetworkEndpointGroupsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroupAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkEndpointGroups.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/networkEndpointGroups', request_field='', request_type_name='ComputeNetworkEndpointGroupsAggregatedListRequest', response_type_name='NetworkEndpointGroupAggregatedList', supports_download=False)

    def AttachNetworkEndpoints(self, request, global_params=None):
        """Attach a list of network endpoints to the specified network endpoint group.

      Args:
        request: (ComputeNetworkEndpointGroupsAttachNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AttachNetworkEndpoints')
        return self._RunMethod(config, request, global_params=global_params)
    AttachNetworkEndpoints.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkEndpointGroups.attachNetworkEndpoints', ordered_params=['project', 'zone', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/networkEndpointGroups/{networkEndpointGroup}/attachNetworkEndpoints', request_field='networkEndpointGroupsAttachEndpointsRequest', request_type_name='ComputeNetworkEndpointGroupsAttachNetworkEndpointsRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified network endpoint group. The network endpoints in the NEG and the VM instances they belong to are not terminated when the NEG is deleted. Note that the NEG cannot be deleted if there are backend services referencing it.

      Args:
        request: (ComputeNetworkEndpointGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.networkEndpointGroups.delete', ordered_params=['project', 'zone', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/networkEndpointGroups/{networkEndpointGroup}', request_field='', request_type_name='ComputeNetworkEndpointGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def DetachNetworkEndpoints(self, request, global_params=None):
        """Detach a list of network endpoints from the specified network endpoint group.

      Args:
        request: (ComputeNetworkEndpointGroupsDetachNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DetachNetworkEndpoints')
        return self._RunMethod(config, request, global_params=global_params)
    DetachNetworkEndpoints.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkEndpointGroups.detachNetworkEndpoints', ordered_params=['project', 'zone', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/networkEndpointGroups/{networkEndpointGroup}/detachNetworkEndpoints', request_field='networkEndpointGroupsDetachEndpointsRequest', request_type_name='ComputeNetworkEndpointGroupsDetachNetworkEndpointsRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified network endpoint group.

      Args:
        request: (ComputeNetworkEndpointGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkEndpointGroups.get', ordered_params=['project', 'zone', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/networkEndpointGroups/{networkEndpointGroup}', request_field='', request_type_name='ComputeNetworkEndpointGroupsGetRequest', response_type_name='NetworkEndpointGroup', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a network endpoint group in the specified project using the parameters that are included in the request.

      Args:
        request: (ComputeNetworkEndpointGroupsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkEndpointGroups.insert', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/networkEndpointGroups', request_field='networkEndpointGroup', request_type_name='ComputeNetworkEndpointGroupsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of network endpoint groups that are located in the specified project and zone.

      Args:
        request: (ComputeNetworkEndpointGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroupList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkEndpointGroups.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/networkEndpointGroups', request_field='', request_type_name='ComputeNetworkEndpointGroupsListRequest', response_type_name='NetworkEndpointGroupList', supports_download=False)

    def ListNetworkEndpoints(self, request, global_params=None):
        """Lists the network endpoints in the specified network endpoint group.

      Args:
        request: (ComputeNetworkEndpointGroupsListNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroupsListNetworkEndpoints) The response message.
      """
        config = self.GetMethodConfig('ListNetworkEndpoints')
        return self._RunMethod(config, request, global_params=global_params)
    ListNetworkEndpoints.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkEndpointGroups.listNetworkEndpoints', ordered_params=['project', 'zone', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/networkEndpointGroups/{networkEndpointGroup}/listNetworkEndpoints', request_field='networkEndpointGroupsListEndpointsRequest', request_type_name='ComputeNetworkEndpointGroupsListNetworkEndpointsRequest', response_type_name='NetworkEndpointGroupsListNetworkEndpoints', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeNetworkEndpointGroupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkEndpointGroups.testIamPermissions', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/networkEndpointGroups/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeNetworkEndpointGroupsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)