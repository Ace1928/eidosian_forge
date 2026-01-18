from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RoutersService(base_api.BaseApiService):
    """Service class for the routers resource."""
    _NAME = 'routers'

    def __init__(self, client):
        super(ComputeBeta.RoutersService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of routers. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeRoutersAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RouterAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routers.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/routers', request_field='', request_type_name='ComputeRoutersAggregatedListRequest', response_type_name='RouterAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified Router resource.

      Args:
        request: (ComputeRoutersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.routers.delete', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/routers/{router}', request_field='', request_type_name='ComputeRoutersDeleteRequest', response_type_name='Operation', supports_download=False)

    def DeleteRoutePolicy(self, request, global_params=None):
        """Deletes Route Policy.

      Args:
        request: (ComputeRoutersDeleteRoutePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DeleteRoutePolicy')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteRoutePolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.routers.deleteRoutePolicy', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['policy', 'requestId'], relative_path='projects/{project}/regions/{region}/routers/{router}/deleteRoutePolicy', request_field='', request_type_name='ComputeRoutersDeleteRoutePolicyRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified Router resource.

      Args:
        request: (ComputeRoutersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Router) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routers.get', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=[], relative_path='projects/{project}/regions/{region}/routers/{router}', request_field='', request_type_name='ComputeRoutersGetRequest', response_type_name='Router', supports_download=False)

    def GetNatIpInfo(self, request, global_params=None):
        """Retrieves runtime NAT IP information.

      Args:
        request: (ComputeRoutersGetNatIpInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NatIpInfoResponse) The response message.
      """
        config = self.GetMethodConfig('GetNatIpInfo')
        return self._RunMethod(config, request, global_params=global_params)
    GetNatIpInfo.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routers.getNatIpInfo', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['natName'], relative_path='projects/{project}/regions/{region}/routers/{router}/getNatIpInfo', request_field='', request_type_name='ComputeRoutersGetNatIpInfoRequest', response_type_name='NatIpInfoResponse', supports_download=False)

    def GetNatMappingInfo(self, request, global_params=None):
        """Retrieves runtime Nat mapping information of VM endpoints.

      Args:
        request: (ComputeRoutersGetNatMappingInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VmEndpointNatMappingsList) The response message.
      """
        config = self.GetMethodConfig('GetNatMappingInfo')
        return self._RunMethod(config, request, global_params=global_params)
    GetNatMappingInfo.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routers.getNatMappingInfo', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['filter', 'maxResults', 'natName', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/routers/{router}/getNatMappingInfo', request_field='', request_type_name='ComputeRoutersGetNatMappingInfoRequest', response_type_name='VmEndpointNatMappingsList', supports_download=False)

    def GetRoutePolicy(self, request, global_params=None):
        """Returns specified Route Policy.

      Args:
        request: (ComputeRoutersGetRoutePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RoutersGetRoutePolicyResponse) The response message.
      """
        config = self.GetMethodConfig('GetRoutePolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetRoutePolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routers.getRoutePolicy', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['policy'], relative_path='projects/{project}/regions/{region}/routers/{router}/getRoutePolicy', request_field='', request_type_name='ComputeRoutersGetRoutePolicyRequest', response_type_name='RoutersGetRoutePolicyResponse', supports_download=False)

    def GetRouterStatus(self, request, global_params=None):
        """Retrieves runtime information of the specified router.

      Args:
        request: (ComputeRoutersGetRouterStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RouterStatusResponse) The response message.
      """
        config = self.GetMethodConfig('GetRouterStatus')
        return self._RunMethod(config, request, global_params=global_params)
    GetRouterStatus.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routers.getRouterStatus', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=[], relative_path='projects/{project}/regions/{region}/routers/{router}/getRouterStatus', request_field='', request_type_name='ComputeRoutersGetRouterStatusRequest', response_type_name='RouterStatusResponse', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a Router resource in the specified project and region using the data included in the request.

      Args:
        request: (ComputeRoutersInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.routers.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/routers', request_field='router', request_type_name='ComputeRoutersInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of Router resources available to the specified project.

      Args:
        request: (ComputeRoutersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RouterList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routers.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/routers', request_field='', request_type_name='ComputeRoutersListRequest', response_type_name='RouterList', supports_download=False)

    def ListBgpRoutes(self, request, global_params=None):
        """Retrieves a list of router bgp routes available to the specified project.

      Args:
        request: (ComputeRoutersListBgpRoutesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RoutersListBgpRoutes) The response message.
      """
        config = self.GetMethodConfig('ListBgpRoutes')
        return self._RunMethod(config, request, global_params=global_params)
    ListBgpRoutes.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routers.listBgpRoutes', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['addressFamily', 'destinationPrefix', 'filter', 'maxResults', 'orderBy', 'pageToken', 'peer', 'policyApplied', 'returnPartialSuccess', 'routeType'], relative_path='projects/{project}/regions/{region}/routers/{router}/listBgpRoutes', request_field='', request_type_name='ComputeRoutersListBgpRoutesRequest', response_type_name='RoutersListBgpRoutes', supports_download=False)

    def ListRoutePolicies(self, request, global_params=None):
        """Retrieves a list of router route policy subresources available to the specified project.

      Args:
        request: (ComputeRoutersListRoutePoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RoutersListRoutePolicies) The response message.
      """
        config = self.GetMethodConfig('ListRoutePolicies')
        return self._RunMethod(config, request, global_params=global_params)
    ListRoutePolicies.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.routers.listRoutePolicies', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/routers/{router}/listRoutePolicies', request_field='', request_type_name='ComputeRoutersListRoutePoliciesRequest', response_type_name='RoutersListRoutePolicies', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified Router resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputeRoutersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.routers.patch', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/routers/{router}', request_field='routerResource', request_type_name='ComputeRoutersPatchRequest', response_type_name='Operation', supports_download=False)

    def Preview(self, request, global_params=None):
        """Preview fields auto-generated during router create and update operations. Calling this method does NOT create or update the router.

      Args:
        request: (ComputeRoutersPreviewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RoutersPreviewResponse) The response message.
      """
        config = self.GetMethodConfig('Preview')
        return self._RunMethod(config, request, global_params=global_params)
    Preview.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.routers.preview', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=[], relative_path='projects/{project}/regions/{region}/routers/{router}/preview', request_field='routerResource', request_type_name='ComputeRoutersPreviewRequest', response_type_name='RoutersPreviewResponse', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRoutersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.routers.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/routers/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRoutersTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified Router resource with the data included in the request. This method conforms to PUT semantics, which requests that the state of the target resource be created or replaced with the state defined by the representation enclosed in the request message payload.

      Args:
        request: (ComputeRoutersUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.routers.update', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/routers/{router}', request_field='routerResource', request_type_name='ComputeRoutersUpdateRequest', response_type_name='Operation', supports_download=False)

    def UpdateRoutePolicy(self, request, global_params=None):
        """Updates or creates new Route Policy.

      Args:
        request: (ComputeRoutersUpdateRoutePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateRoutePolicy')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateRoutePolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.routers.updateRoutePolicy', ordered_params=['project', 'region', 'router'], path_params=['project', 'region', 'router'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/routers/{router}/updateRoutePolicy', request_field='routePolicy', request_type_name='ComputeRoutersUpdateRoutePolicyRequest', response_type_name='Operation', supports_download=False)