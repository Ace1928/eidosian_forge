from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1 import networkconnectivity_v1_messages as messages
class ProjectsLocationsServiceConnectionMapsService(base_api.BaseApiService):
    """Service class for the projects_locations_serviceConnectionMaps resource."""
    _NAME = 'projects_locations_serviceConnectionMaps'

    def __init__(self, client):
        super(NetworkconnectivityV1.ProjectsLocationsServiceConnectionMapsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ServiceConnectionMap in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionMapsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionMaps', http_method='POST', method_id='networkconnectivity.projects.locations.serviceConnectionMaps.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'serviceConnectionMapId'], relative_path='v1/{+parent}/serviceConnectionMaps', request_field='serviceConnectionMap', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionMapsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ServiceConnectionMap.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionMapsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionMaps/{serviceConnectionMapsId}', http_method='DELETE', method_id='networkconnectivity.projects.locations.serviceConnectionMaps.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionMapsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ServiceConnectionMap.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionMapsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceConnectionMap) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionMaps/{serviceConnectionMapsId}', http_method='GET', method_id='networkconnectivity.projects.locations.serviceConnectionMaps.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionMapsGetRequest', response_type_name='ServiceConnectionMap', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionMapsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionMaps/{serviceConnectionMapsId}:getIamPolicy', http_method='GET', method_id='networkconnectivity.projects.locations.serviceConnectionMaps.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionMapsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ServiceConnectionMaps in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionMapsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceConnectionMapsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionMaps', http_method='GET', method_id='networkconnectivity.projects.locations.serviceConnectionMaps.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/serviceConnectionMaps', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionMapsListRequest', response_type_name='ListServiceConnectionMapsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ServiceConnectionMap.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionMapsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionMaps/{serviceConnectionMapsId}', http_method='PATCH', method_id='networkconnectivity.projects.locations.serviceConnectionMaps.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='serviceConnectionMap', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionMapsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionMapsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionMaps/{serviceConnectionMapsId}:setIamPolicy', http_method='POST', method_id='networkconnectivity.projects.locations.serviceConnectionMaps.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionMapsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionMapsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionMaps/{serviceConnectionMapsId}:testIamPermissions', http_method='POST', method_id='networkconnectivity.projects.locations.serviceConnectionMaps.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionMapsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)