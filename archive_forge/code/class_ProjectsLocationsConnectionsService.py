from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
class ProjectsLocationsConnectionsService(base_api.BaseApiService):
    """Service class for the projects_locations_connections resource."""
    _NAME = 'projects_locations_connections'

    def __init__(self, client):
        super(CloudbuildV2.ProjectsLocationsConnectionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Connection.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections', http_method='POST', method_id='cloudbuild.projects.locations.connections.create', ordered_params=['parent'], path_params=['parent'], query_params=['connectionId'], relative_path='v2/{+parent}/connections', request_field='connection', request_type_name='CloudbuildProjectsLocationsConnectionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single connection.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}', http_method='DELETE', method_id='cloudbuild.projects.locations.connections.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v2/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsConnectionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def FetchLinkableRepositories(self, request, global_params=None):
        """FetchLinkableRepositories get repositories from SCM that are accessible and could be added to the connection.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsFetchLinkableRepositoriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchLinkableRepositoriesResponse) The response message.
      """
        config = self.GetMethodConfig('FetchLinkableRepositories')
        return self._RunMethod(config, request, global_params=global_params)
    FetchLinkableRepositories.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}:fetchLinkableRepositories', http_method='GET', method_id='cloudbuild.projects.locations.connections.fetchLinkableRepositories', ordered_params=['connection'], path_params=['connection'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+connection}:fetchLinkableRepositories', request_field='', request_type_name='CloudbuildProjectsLocationsConnectionsFetchLinkableRepositoriesRequest', response_type_name='FetchLinkableRepositoriesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single connection.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Connection) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}', http_method='GET', method_id='cloudbuild.projects.locations.connections.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsConnectionsGetRequest', response_type_name='Connection', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}:getIamPolicy', http_method='GET', method_id='cloudbuild.projects.locations.connections.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v2/{+resource}:getIamPolicy', request_field='', request_type_name='CloudbuildProjectsLocationsConnectionsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Connections in a given project and location.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConnectionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections', http_method='GET', method_id='cloudbuild.projects.locations.connections.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/connections', request_field='', request_type_name='CloudbuildProjectsLocationsConnectionsListRequest', response_type_name='ListConnectionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a single connection.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.connections.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'updateMask'], relative_path='v2/{+name}', request_field='connection', request_type_name='CloudbuildProjectsLocationsConnectionsPatchRequest', response_type_name='Operation', supports_download=False)

    def ProcessWebhook(self, request, global_params=None):
        """ProcessWebhook is called by the external SCM for notifying of events.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsProcessWebhookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('ProcessWebhook')
        return self._RunMethod(config, request, global_params=global_params)
    ProcessWebhook.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections:processWebhook', http_method='POST', method_id='cloudbuild.projects.locations.connections.processWebhook', ordered_params=['parent'], path_params=['parent'], query_params=['webhookKey'], relative_path='v2/{+parent}/connections:processWebhook', request_field='httpBody', request_type_name='CloudbuildProjectsLocationsConnectionsProcessWebhookRequest', response_type_name='Empty', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}:setIamPolicy', http_method='POST', method_id='cloudbuild.projects.locations.connections.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='CloudbuildProjectsLocationsConnectionsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}:testIamPermissions', http_method='POST', method_id='cloudbuild.projects.locations.connections.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='CloudbuildProjectsLocationsConnectionsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)