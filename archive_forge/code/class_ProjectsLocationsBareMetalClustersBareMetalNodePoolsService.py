from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class ProjectsLocationsBareMetalClustersBareMetalNodePoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_bareMetalClusters_bareMetalNodePools resource."""
    _NAME = 'projects_locations_bareMetalClusters_bareMetalNodePools'

    def __init__(self, client):
        super(GkeonpremV1.ProjectsLocationsBareMetalClustersBareMetalNodePoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new bare metal node pool in a given project, location and Bare Metal cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.create', ordered_params=['parent'], path_params=['parent'], query_params=['bareMetalNodePoolId', 'validateOnly'], relative_path='v1/{+parent}/bareMetalNodePools', request_field='bareMetalNodePool', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single bare metal node pool.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools/{bareMetalNodePoolsId}', http_method='DELETE', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'ignoreErrors', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Enroll(self, request, global_params=None):
        """Enrolls an existing bare metal node pool to the Anthos On-Prem API within a given project and location. Through enrollment, an existing node pool will become Anthos On-Prem API managed. The corresponding GCP resources will be created.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsEnrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Enroll')
        return self._RunMethod(config, request, global_params=global_params)
    Enroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools:enroll', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.enroll', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/bareMetalNodePools:enroll', request_field='enrollBareMetalNodePoolRequest', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsEnrollRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single bare metal node pool.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BareMetalNodePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools/{bareMetalNodePoolsId}', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsGetRequest', response_type_name='BareMetalNodePool', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools/{bareMetalNodePoolsId}:getIamPolicy', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists bare metal node pools in a given project, location and bare metal cluster.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBareMetalNodePoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools', http_method='GET', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/bareMetalNodePools', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsListRequest', response_type_name='ListBareMetalNodePoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single bare metal node pool.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools/{bareMetalNodePoolsId}', http_method='PATCH', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='bareMetalNodePool', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools/{bareMetalNodePoolsId}:setIamPolicy', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools/{bareMetalNodePoolsId}:testIamPermissions', http_method='POST', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Unenroll(self, request, global_params=None):
        """Unenrolls a bare metal node pool from Anthos On-Prem API.

      Args:
        request: (GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsUnenrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Unenroll')
        return self._RunMethod(config, request, global_params=global_params)
    Unenroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/bareMetalClusters/{bareMetalClustersId}/bareMetalNodePools/{bareMetalNodePoolsId}:unenroll', http_method='DELETE', method_id='gkeonprem.projects.locations.bareMetalClusters.bareMetalNodePools.unenroll', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'validateOnly'], relative_path='v1/{+name}:unenroll', request_field='', request_type_name='GkeonpremProjectsLocationsBareMetalClustersBareMetalNodePoolsUnenrollRequest', response_type_name='Operation', supports_download=False)