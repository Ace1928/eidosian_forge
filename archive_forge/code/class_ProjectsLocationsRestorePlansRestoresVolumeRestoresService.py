from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkebackup.v1 import gkebackup_v1_messages as messages
class ProjectsLocationsRestorePlansRestoresVolumeRestoresService(base_api.BaseApiService):
    """Service class for the projects_locations_restorePlans_restores_volumeRestores resource."""
    _NAME = 'projects_locations_restorePlans_restores_volumeRestores'

    def __init__(self, client):
        super(GkebackupV1.ProjectsLocationsRestorePlansRestoresVolumeRestoresService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieve the details of a single VolumeRestore.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VolumeRestore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}/volumeRestores/{volumeRestoresId}', http_method='GET', method_id='gkebackup.projects.locations.restorePlans.restores.volumeRestores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresGetRequest', response_type_name='VolumeRestore', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}/volumeRestores/{volumeRestoresId}:getIamPolicy', http_method='GET', method_id='gkebackup.projects.locations.restorePlans.restores.volumeRestores.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the VolumeRestores for a given Restore.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVolumeRestoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}/volumeRestores', http_method='GET', method_id='gkebackup.projects.locations.restorePlans.restores.volumeRestores.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/volumeRestores', request_field='', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresListRequest', response_type_name='ListVolumeRestoresResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}/volumeRestores/{volumeRestoresId}:setIamPolicy', http_method='POST', method_id='gkebackup.projects.locations.restorePlans.restores.volumeRestores.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/restorePlans/{restorePlansId}/restores/{restoresId}/volumeRestores/{volumeRestoresId}:testIamPermissions', http_method='POST', method_id='gkebackup.projects.locations.restorePlans.restores.volumeRestores.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkebackupProjectsLocationsRestorePlansRestoresVolumeRestoresTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)