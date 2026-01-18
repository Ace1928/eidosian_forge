from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class ProjectsLocationsScopesService(base_api.BaseApiService):
    """Service class for the projects_locations_scopes resource."""
    _NAME = 'projects_locations_scopes'

    def __init__(self, client):
        super(GkehubV1beta.ProjectsLocationsScopesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Scope.

      Args:
        request: (GkehubProjectsLocationsScopesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes', http_method='POST', method_id='gkehub.projects.locations.scopes.create', ordered_params=['parent'], path_params=['parent'], query_params=['scopeId'], relative_path='v1beta/{+parent}/scopes', request_field='scope', request_type_name='GkehubProjectsLocationsScopesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Scope.

      Args:
        request: (GkehubProjectsLocationsScopesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}', http_method='DELETE', method_id='gkehub.projects.locations.scopes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsScopesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the details of a Scope.

      Args:
        request: (GkehubProjectsLocationsScopesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Scope) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}', http_method='GET', method_id='gkehub.projects.locations.scopes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsScopesGetRequest', response_type_name='Scope', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkehubProjectsLocationsScopesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}:getIamPolicy', http_method='GET', method_id='gkehub.projects.locations.scopes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta/{+resource}:getIamPolicy', request_field='', request_type_name='GkehubProjectsLocationsScopesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Scopes.

      Args:
        request: (GkehubProjectsLocationsScopesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListScopesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes', http_method='GET', method_id='gkehub.projects.locations.scopes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/scopes', request_field='', request_type_name='GkehubProjectsLocationsScopesListRequest', response_type_name='ListScopesResponse', supports_download=False)

    def ListMemberships(self, request, global_params=None):
        """Lists Memberships bound to a Scope. The response includes relevant Memberships from all regions.

      Args:
        request: (GkehubProjectsLocationsScopesListMembershipsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBoundMembershipsResponse) The response message.
      """
        config = self.GetMethodConfig('ListMemberships')
        return self._RunMethod(config, request, global_params=global_params)
    ListMemberships.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}:listMemberships', http_method='GET', method_id='gkehub.projects.locations.scopes.listMemberships', ordered_params=['scopeName'], path_params=['scopeName'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta/{+scopeName}:listMemberships', request_field='', request_type_name='GkehubProjectsLocationsScopesListMembershipsRequest', response_type_name='ListBoundMembershipsResponse', supports_download=False)

    def ListPermitted(self, request, global_params=None):
        """Lists permitted Scopes.

      Args:
        request: (GkehubProjectsLocationsScopesListPermittedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPermittedScopesResponse) The response message.
      """
        config = self.GetMethodConfig('ListPermitted')
        return self._RunMethod(config, request, global_params=global_params)
    ListPermitted.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes:listPermitted', http_method='GET', method_id='gkehub.projects.locations.scopes.listPermitted', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/scopes:listPermitted', request_field='', request_type_name='GkehubProjectsLocationsScopesListPermittedRequest', response_type_name='ListPermittedScopesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a scopes.

      Args:
        request: (GkehubProjectsLocationsScopesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}', http_method='PATCH', method_id='gkehub.projects.locations.scopes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='scope', request_type_name='GkehubProjectsLocationsScopesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkehubProjectsLocationsScopesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}:setIamPolicy', http_method='POST', method_id='gkehub.projects.locations.scopes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkehubProjectsLocationsScopesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkehubProjectsLocationsScopesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}:testIamPermissions', http_method='POST', method_id='gkehub.projects.locations.scopes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkehubProjectsLocationsScopesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)