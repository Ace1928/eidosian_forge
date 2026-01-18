from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1beta import networkconnectivity_v1beta_messages as messages
class ProjectsLocationsSpokesService(base_api.BaseApiService):
    """Service class for the projects_locations_spokes resource."""
    _NAME = 'projects_locations_spokes'

    def __init__(self, client):
        super(NetworkconnectivityV1beta.ProjectsLocationsSpokesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Network Connectivity Center spoke.

      Args:
        request: (NetworkconnectivityProjectsLocationsSpokesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/spokes', http_method='POST', method_id='networkconnectivity.projects.locations.spokes.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'spokeId'], relative_path='v1beta/{+parent}/spokes', request_field='googleCloudNetworkconnectivityV1betaSpoke', request_type_name='NetworkconnectivityProjectsLocationsSpokesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Network Connectivity Center spoke.

      Args:
        request: (NetworkconnectivityProjectsLocationsSpokesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/spokes/{spokesId}', http_method='DELETE', method_id='networkconnectivity.projects.locations.spokes.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsSpokesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a Network Connectivity Center spoke.

      Args:
        request: (NetworkconnectivityProjectsLocationsSpokesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaSpoke) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/spokes/{spokesId}', http_method='GET', method_id='networkconnectivity.projects.locations.spokes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsSpokesGetRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaSpoke', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkconnectivityProjectsLocationsSpokesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/spokes/{spokesId}:getIamPolicy', http_method='GET', method_id='networkconnectivity.projects.locations.spokes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkconnectivityProjectsLocationsSpokesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Network Connectivity Center spokes in a specified project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsSpokesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaListSpokesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/spokes', http_method='GET', method_id='networkconnectivity.projects.locations.spokes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/spokes', request_field='', request_type_name='NetworkconnectivityProjectsLocationsSpokesListRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaListSpokesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a Network Connectivity Center spoke.

      Args:
        request: (NetworkconnectivityProjectsLocationsSpokesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/spokes/{spokesId}', http_method='PATCH', method_id='networkconnectivity.projects.locations.spokes.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1beta/{+name}', request_field='googleCloudNetworkconnectivityV1betaSpoke', request_type_name='NetworkconnectivityProjectsLocationsSpokesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkconnectivityProjectsLocationsSpokesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/spokes/{spokesId}:setIamPolicy', http_method='POST', method_id='networkconnectivity.projects.locations.spokes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='NetworkconnectivityProjectsLocationsSpokesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkconnectivityProjectsLocationsSpokesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/spokes/{spokesId}:testIamPermissions', http_method='POST', method_id='networkconnectivity.projects.locations.spokes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='NetworkconnectivityProjectsLocationsSpokesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)