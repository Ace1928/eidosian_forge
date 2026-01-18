from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sddc.v1alpha1 import sddc_v1alpha1_messages as messages
class ProjectsLocationsClusterGroupsService(base_api.BaseApiService):
    """Service class for the projects_locations_clusterGroups resource."""
    _NAME = 'projects_locations_clusterGroups'

    def __init__(self, client):
        super(SddcV1alpha1.ProjectsLocationsClusterGroupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `ClusterGroup` in a given project and location (region). The creation is asynchronous. You can check the returned operation to track its progress. When the operation successfully completes, the new `ClusterGroup` is fully functional. The returned operation is automatically deleted after a few hours, so there is no need to call `DeleteOperation`.

      Args:
        request: (SddcProjectsLocationsClusterGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups', http_method='POST', method_id='sddc.projects.locations.clusterGroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['clusterGroupId'], relative_path='v1alpha1/{+parent}/clusterGroups', request_field='clusterGroup', request_type_name='SddcProjectsLocationsClusterGroupsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `ClusterGroup`.

      Args:
        request: (SddcProjectsLocationsClusterGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}', http_method='DELETE', method_id='sddc.projects.locations.clusterGroups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def GenerateSupportBundle(self, request, global_params=None):
        """Consumer API (private) to generate support bundles of VMware stack.

      Args:
        request: (SddcProjectsLocationsClusterGroupsGenerateSupportBundleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('GenerateSupportBundle')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateSupportBundle.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}:generateSupportBundle', http_method='POST', method_id='sddc.projects.locations.clusterGroups.generateSupportBundle', ordered_params=['clusterGroup'], path_params=['clusterGroup'], query_params=[], relative_path='v1alpha1/{+clusterGroup}:generateSupportBundle', request_field='generateSupportBundleRequest', request_type_name='SddcProjectsLocationsClusterGroupsGenerateSupportBundleRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single `ClusterGroup`.

      Args:
        request: (SddcProjectsLocationsClusterGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ClusterGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}', http_method='GET', method_id='sddc.projects.locations.clusterGroups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsGetRequest', response_type_name='ClusterGroup', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (SddcProjectsLocationsClusterGroupsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}:getIamPolicy', http_method='GET', method_id='sddc.projects.locations.clusterGroups.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha1/{+resource}:getIamPolicy', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `ClusterGroup` objects in a given project and location (region).

      Args:
        request: (SddcProjectsLocationsClusterGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListClusterGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups', http_method='GET', method_id='sddc.projects.locations.clusterGroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/clusterGroups', request_field='', request_type_name='SddcProjectsLocationsClusterGroupsListRequest', response_type_name='ListClusterGroupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the description, labels, and `NetworkConfig` of a specific `ClusterGroup`.

      Args:
        request: (SddcProjectsLocationsClusterGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}', http_method='PATCH', method_id='sddc.projects.locations.clusterGroups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='clusterGroup', request_type_name='SddcProjectsLocationsClusterGroupsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ResetCloudAdminCredentials(self, request, global_params=None):
        """Reset the vCenter or NSX cloudadmin accounts.

      Args:
        request: (SddcProjectsLocationsClusterGroupsResetCloudAdminCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ResetCloudAdminCredentials')
        return self._RunMethod(config, request, global_params=global_params)
    ResetCloudAdminCredentials.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}:resetCloudAdminCredentials', http_method='POST', method_id='sddc.projects.locations.clusterGroups.resetCloudAdminCredentials', ordered_params=['clusterGroup'], path_params=['clusterGroup'], query_params=[], relative_path='v1alpha1/{+clusterGroup}:resetCloudAdminCredentials', request_field='resetCloudAdminCredentialsRequest', request_type_name='SddcProjectsLocationsClusterGroupsResetCloudAdminCredentialsRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (SddcProjectsLocationsClusterGroupsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}:setIamPolicy', http_method='POST', method_id='sddc.projects.locations.clusterGroups.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='SddcProjectsLocationsClusterGroupsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (SddcProjectsLocationsClusterGroupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/clusterGroups/{clusterGroupsId}:testIamPermissions', http_method='POST', method_id='sddc.projects.locations.clusterGroups.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='SddcProjectsLocationsClusterGroupsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)