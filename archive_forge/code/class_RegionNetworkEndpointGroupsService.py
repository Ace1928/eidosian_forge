from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionNetworkEndpointGroupsService(base_api.BaseApiService):
    """Service class for the regionNetworkEndpointGroups resource."""
    _NAME = 'regionNetworkEndpointGroups'

    def __init__(self, client):
        super(ComputeBeta.RegionNetworkEndpointGroupsService, self).__init__(client)
        self._upload_configs = {}

    def AttachNetworkEndpoints(self, request, global_params=None):
        """Attach a list of network endpoints to the specified network endpoint group.

      Args:
        request: (ComputeRegionNetworkEndpointGroupsAttachNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AttachNetworkEndpoints')
        return self._RunMethod(config, request, global_params=global_params)
    AttachNetworkEndpoints.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkEndpointGroups.attachNetworkEndpoints', ordered_params=['project', 'region', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/networkEndpointGroups/{networkEndpointGroup}/attachNetworkEndpoints', request_field='regionNetworkEndpointGroupsAttachEndpointsRequest', request_type_name='ComputeRegionNetworkEndpointGroupsAttachNetworkEndpointsRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified network endpoint group. Note that the NEG cannot be deleted if it is configured as a backend of a backend service.

      Args:
        request: (ComputeRegionNetworkEndpointGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionNetworkEndpointGroups.delete', ordered_params=['project', 'region', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/networkEndpointGroups/{networkEndpointGroup}', request_field='', request_type_name='ComputeRegionNetworkEndpointGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def DetachNetworkEndpoints(self, request, global_params=None):
        """Detach the network endpoint from the specified network endpoint group.

      Args:
        request: (ComputeRegionNetworkEndpointGroupsDetachNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DetachNetworkEndpoints')
        return self._RunMethod(config, request, global_params=global_params)
    DetachNetworkEndpoints.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkEndpointGroups.detachNetworkEndpoints', ordered_params=['project', 'region', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/networkEndpointGroups/{networkEndpointGroup}/detachNetworkEndpoints', request_field='regionNetworkEndpointGroupsDetachEndpointsRequest', request_type_name='ComputeRegionNetworkEndpointGroupsDetachNetworkEndpointsRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified network endpoint group.

      Args:
        request: (ComputeRegionNetworkEndpointGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNetworkEndpointGroups.get', ordered_params=['project', 'region', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/networkEndpointGroups/{networkEndpointGroup}', request_field='', request_type_name='ComputeRegionNetworkEndpointGroupsGetRequest', response_type_name='NetworkEndpointGroup', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a network endpoint group in the specified project using the parameters that are included in the request.

      Args:
        request: (ComputeRegionNetworkEndpointGroupsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkEndpointGroups.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/networkEndpointGroups', request_field='networkEndpointGroup', request_type_name='ComputeRegionNetworkEndpointGroupsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of regional network endpoint groups available to the specified project in the given region.

      Args:
        request: (ComputeRegionNetworkEndpointGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroupList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNetworkEndpointGroups.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/networkEndpointGroups', request_field='', request_type_name='ComputeRegionNetworkEndpointGroupsListRequest', response_type_name='NetworkEndpointGroupList', supports_download=False)

    def ListNetworkEndpoints(self, request, global_params=None):
        """Lists the network endpoints in the specified network endpoint group.

      Args:
        request: (ComputeRegionNetworkEndpointGroupsListNetworkEndpointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkEndpointGroupsListNetworkEndpoints) The response message.
      """
        config = self.GetMethodConfig('ListNetworkEndpoints')
        return self._RunMethod(config, request, global_params=global_params)
    ListNetworkEndpoints.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkEndpointGroups.listNetworkEndpoints', ordered_params=['project', 'region', 'networkEndpointGroup'], path_params=['networkEndpointGroup', 'project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/networkEndpointGroups/{networkEndpointGroup}/listNetworkEndpoints', request_field='', request_type_name='ComputeRegionNetworkEndpointGroupsListNetworkEndpointsRequest', response_type_name='NetworkEndpointGroupsListNetworkEndpoints', supports_download=False)