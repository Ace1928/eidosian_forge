from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.krmapihosting.v1 import krmapihosting_v1_messages as messages
class ProjectsLocationsKrmApiHostsService(base_api.BaseApiService):
    """Service class for the projects_locations_krmApiHosts resource."""
    _NAME = 'projects_locations_krmApiHosts'

    def __init__(self, client):
        super(KrmapihostingV1.ProjectsLocationsKrmApiHostsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new KrmApiHost in a given project and location.

      Args:
        request: (KrmapihostingProjectsLocationsKrmApiHostsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/krmApiHosts', http_method='POST', method_id='krmapihosting.projects.locations.krmApiHosts.create', ordered_params=['parent'], path_params=['parent'], query_params=['krmApiHostId', 'requestId'], relative_path='v1/{+parent}/krmApiHosts', request_field='krmApiHost', request_type_name='KrmapihostingProjectsLocationsKrmApiHostsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single KrmApiHost.

      Args:
        request: (KrmapihostingProjectsLocationsKrmApiHostsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/krmApiHosts/{krmApiHostsId}', http_method='DELETE', method_id='krmapihosting.projects.locations.krmApiHosts.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='KrmapihostingProjectsLocationsKrmApiHostsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single KrmApiHost.

      Args:
        request: (KrmapihostingProjectsLocationsKrmApiHostsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (KrmApiHost) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/krmApiHosts/{krmApiHostsId}', http_method='GET', method_id='krmapihosting.projects.locations.krmApiHosts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='KrmapihostingProjectsLocationsKrmApiHostsGetRequest', response_type_name='KrmApiHost', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (KrmapihostingProjectsLocationsKrmApiHostsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/krmApiHosts/{krmApiHostsId}:getIamPolicy', http_method='GET', method_id='krmapihosting.projects.locations.krmApiHosts.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='KrmapihostingProjectsLocationsKrmApiHostsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists KrmApiHosts in a given project and location.

      Args:
        request: (KrmapihostingProjectsLocationsKrmApiHostsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListKrmApiHostsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/krmApiHosts', http_method='GET', method_id='krmapihosting.projects.locations.krmApiHosts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/krmApiHosts', request_field='', request_type_name='KrmapihostingProjectsLocationsKrmApiHostsListRequest', response_type_name='ListKrmApiHostsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single KrmApiHost.

      Args:
        request: (KrmapihostingProjectsLocationsKrmApiHostsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/krmApiHosts/{krmApiHostsId}', http_method='PATCH', method_id='krmapihosting.projects.locations.krmApiHosts.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='krmApiHost', request_type_name='KrmapihostingProjectsLocationsKrmApiHostsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (KrmapihostingProjectsLocationsKrmApiHostsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/krmApiHosts/{krmApiHostsId}:setIamPolicy', http_method='POST', method_id='krmapihosting.projects.locations.krmApiHosts.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='KrmapihostingProjectsLocationsKrmApiHostsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (KrmapihostingProjectsLocationsKrmApiHostsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/krmApiHosts/{krmApiHostsId}:testIamPermissions', http_method='POST', method_id='krmapihosting.projects.locations.krmApiHosts.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='KrmapihostingProjectsLocationsKrmApiHostsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)