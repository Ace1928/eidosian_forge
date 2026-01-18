from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkebackup.v1 import gkebackup_v1_messages as messages
class ProjectsLocationsRestorePlansRestoresService(base_api.BaseApiService):
    """Service class for the projects_locations_restorePlans_restores resource."""
    _NAME = 'projects_locations_restorePlans_restores'

    def __init__(self, client):
        super(GkebackupV1.ProjectsLocationsRestorePlansRestoresService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Restore for the given RestorePlan.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores', http_method='POST', method_id='gkebackup.projects.locations.restorePlans.restores.create', ordered_params=['parent'], path_params=['parent'], query_params=['restoreId'], relative_path='v1/{+parent}/restores', request_field='restore', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing Restore.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}', http_method='DELETE', method_id='gkebackup.projects.locations.restorePlans.restores.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'force'], relative_path='v1/{+name}', request_field='', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the details of a single Restore.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Restore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}', http_method='GET', method_id='gkebackup.projects.locations.restorePlans.restores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresGetRequest', response_type_name='Restore', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}:getIamPolicy', http_method='GET', method_id='gkebackup.projects.locations.restorePlans.restores.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Restores for a given RestorePlan.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRestoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores', http_method='GET', method_id='gkebackup.projects.locations.restorePlans.restores.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/restores', request_field='', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresListRequest', response_type_name='ListRestoresResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a Restore.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}', http_method='PATCH', method_id='gkebackup.projects.locations.restorePlans.restores.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='restore', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}:setIamPolicy', http_method='POST', method_id='gkebackup.projects.locations.restorePlans.restores.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}:testIamPermissions', http_method='POST', method_id='gkebackup.projects.locations.restorePlans.restores.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)