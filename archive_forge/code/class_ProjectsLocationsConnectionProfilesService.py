from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1alpha2 import datamigration_v1alpha2_messages as messages
class ProjectsLocationsConnectionProfilesService(base_api.BaseApiService):
    """Service class for the projects_locations_connectionProfiles resource."""
    _NAME = 'projects_locations_connectionProfiles'

    def __init__(self, client):
        super(DatamigrationV1alpha2.ProjectsLocationsConnectionProfilesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new connection profile in a given project and location.

      Args:
        request: (DatamigrationProjectsLocationsConnectionProfilesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/connectionProfiles', http_method='POST', method_id='datamigration.projects.locations.connectionProfiles.create', ordered_params=['parent'], path_params=['parent'], query_params=['connectionProfileId', 'requestId'], relative_path='v1alpha2/{+parent}/connectionProfiles', request_field='connectionProfile', request_type_name='DatamigrationProjectsLocationsConnectionProfilesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Database Migration Service connection profile. A connection profile can only be deleted if it is not in use by any active migration jobs.

      Args:
        request: (DatamigrationProjectsLocationsConnectionProfilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/connectionProfiles/{connectionProfilesId}', http_method='DELETE', method_id='datamigration.projects.locations.connectionProfiles.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v1alpha2/{+name}', request_field='', request_type_name='DatamigrationProjectsLocationsConnectionProfilesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single connection profile.

      Args:
        request: (DatamigrationProjectsLocationsConnectionProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConnectionProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/connectionProfiles/{connectionProfilesId}', http_method='GET', method_id='datamigration.projects.locations.connectionProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='DatamigrationProjectsLocationsConnectionProfilesGetRequest', response_type_name='ConnectionProfile', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DatamigrationProjectsLocationsConnectionProfilesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/connectionProfiles/{connectionProfilesId}:getIamPolicy', http_method='GET', method_id='datamigration.projects.locations.connectionProfiles.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='DatamigrationProjectsLocationsConnectionProfilesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieve a list of all connection profiles in a given project and location.

      Args:
        request: (DatamigrationProjectsLocationsConnectionProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConnectionProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/connectionProfiles', http_method='GET', method_id='datamigration.projects.locations.connectionProfiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/connectionProfiles', request_field='', request_type_name='DatamigrationProjectsLocationsConnectionProfilesListRequest', response_type_name='ListConnectionProfilesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update the configuration of a single connection profile.

      Args:
        request: (DatamigrationProjectsLocationsConnectionProfilesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/connectionProfiles/{connectionProfilesId}', http_method='PATCH', method_id='datamigration.projects.locations.connectionProfiles.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha2/{+name}', request_field='connectionProfile', request_type_name='DatamigrationProjectsLocationsConnectionProfilesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (DatamigrationProjectsLocationsConnectionProfilesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/connectionProfiles/{connectionProfilesId}:setIamPolicy', http_method='POST', method_id='datamigration.projects.locations.connectionProfiles.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DatamigrationProjectsLocationsConnectionProfilesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DatamigrationProjectsLocationsConnectionProfilesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/connectionProfiles/{connectionProfilesId}:testIamPermissions', http_method='POST', method_id='datamigration.projects.locations.connectionProfiles.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatamigrationProjectsLocationsConnectionProfilesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)