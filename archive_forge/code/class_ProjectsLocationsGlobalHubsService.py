from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1beta import networkconnectivity_v1beta_messages as messages
class ProjectsLocationsGlobalHubsService(base_api.BaseApiService):
    """Service class for the projects_locations_global_hubs resource."""
    _NAME = 'projects_locations_global_hubs'

    def __init__(self, client):
        super(NetworkconnectivityV1beta.ProjectsLocationsGlobalHubsService, self).__init__(client)
        self._upload_configs = {}

    def AcceptSpoke(self, request, global_params=None):
        """Accepts a proposal to attach a Network Connectivity Center spoke to a hub.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsAcceptSpokeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('AcceptSpoke')
        return self._RunMethod(config, request, global_params=global_params)
    AcceptSpoke.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}:acceptSpoke', http_method='POST', method_id='networkconnectivity.projects.locations.global.hubs.acceptSpoke', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:acceptSpoke', request_field='googleCloudNetworkconnectivityV1betaAcceptHubSpokeRequest', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsAcceptSpokeRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new Network Connectivity Center hub in the specified project.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs', http_method='POST', method_id='networkconnectivity.projects.locations.global.hubs.create', ordered_params=['parent'], path_params=['parent'], query_params=['hubId', 'requestId'], relative_path='v1beta/{+parent}/hubs', request_field='googleCloudNetworkconnectivityV1betaHub', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Network Connectivity Center hub.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}', http_method='DELETE', method_id='networkconnectivity.projects.locations.global.hubs.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a Network Connectivity Center hub.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaHub) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}', http_method='GET', method_id='networkconnectivity.projects.locations.global.hubs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsGetRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaHub', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}:getIamPolicy', http_method='GET', method_id='networkconnectivity.projects.locations.global.hubs.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Network Connectivity Center hubs associated with a given project.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaListHubsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs', http_method='GET', method_id='networkconnectivity.projects.locations.global.hubs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/hubs', request_field='', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsListRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaListHubsResponse', supports_download=False)

    def ListSpokes(self, request, global_params=None):
        """Lists the Network Connectivity Center spokes associated with a specified hub and location. The list includes both spokes that are attached to the hub and spokes that have been proposed but not yet accepted.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsListSpokesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaListHubSpokesResponse) The response message.
      """
        config = self.GetMethodConfig('ListSpokes')
        return self._RunMethod(config, request, global_params=global_params)
    ListSpokes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}:listSpokes', http_method='GET', method_id='networkconnectivity.projects.locations.global.hubs.listSpokes', ordered_params=['name'], path_params=['name'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'spokeLocations', 'view'], relative_path='v1beta/{+name}:listSpokes', request_field='', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsListSpokesRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaListHubSpokesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the description and/or labels of a Network Connectivity Center hub.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}', http_method='PATCH', method_id='networkconnectivity.projects.locations.global.hubs.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1beta/{+name}', request_field='googleCloudNetworkconnectivityV1betaHub', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def RejectSpoke(self, request, global_params=None):
        """Rejects a Network Connectivity Center spoke from being attached to a hub. If the spoke was previously in the `ACTIVE` state, it transitions to the `INACTIVE` state and is no longer able to connect to other spokes that are attached to the hub.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsRejectSpokeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('RejectSpoke')
        return self._RunMethod(config, request, global_params=global_params)
    RejectSpoke.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}:rejectSpoke', http_method='POST', method_id='networkconnectivity.projects.locations.global.hubs.rejectSpoke', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:rejectSpoke', request_field='googleCloudNetworkconnectivityV1betaRejectHubSpokeRequest', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsRejectSpokeRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}:setIamPolicy', http_method='POST', method_id='networkconnectivity.projects.locations.global.hubs.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/global/hubs/{hubsId}:testIamPermissions', http_method='POST', method_id='networkconnectivity.projects.locations.global.hubs.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='NetworkconnectivityProjectsLocationsGlobalHubsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)