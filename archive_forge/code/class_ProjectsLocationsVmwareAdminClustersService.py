from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
class ProjectsLocationsVmwareAdminClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_vmwareAdminClusters resource."""
    _NAME = 'projects_locations_vmwareAdminClusters'

    def __init__(self, client):
        super(GkeonpremV1.ProjectsLocationsVmwareAdminClustersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new VMware admin cluster in a given project and location. The API needs to be combined with creating a bootstrap cluster to work.

      Args:
        request: (GkeonpremProjectsLocationsVmwareAdminClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareAdminClusters', http_method='POST', method_id='gkeonprem.projects.locations.vmwareAdminClusters.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly', 'vmwareAdminClusterId'], relative_path='v1/{+parent}/vmwareAdminClusters', request_field='vmwareAdminCluster', request_type_name='GkeonpremProjectsLocationsVmwareAdminClustersCreateRequest', response_type_name='Operation', supports_download=False)

    def Enroll(self, request, global_params=None):
        """Enrolls an existing VMware admin cluster to the Anthos On-Prem API within a given project and location. Through enrollment, an existing admin cluster will become Anthos On-Prem API managed. The corresponding GCP resources will be created and all future modifications to the cluster will be expected to be performed through the API.

      Args:
        request: (GkeonpremProjectsLocationsVmwareAdminClustersEnrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Enroll')
        return self._RunMethod(config, request, global_params=global_params)
    Enroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareAdminClusters:enroll', http_method='POST', method_id='gkeonprem.projects.locations.vmwareAdminClusters.enroll', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/vmwareAdminClusters:enroll', request_field='enrollVmwareAdminClusterRequest', request_type_name='GkeonpremProjectsLocationsVmwareAdminClustersEnrollRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single VMware admin cluster.

      Args:
        request: (GkeonpremProjectsLocationsVmwareAdminClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VmwareAdminCluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareAdminClusters/{vmwareAdminClustersId}', http_method='GET', method_id='gkeonprem.projects.locations.vmwareAdminClusters.get', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'view'], relative_path='v1/{+name}', request_field='', request_type_name='GkeonpremProjectsLocationsVmwareAdminClustersGetRequest', response_type_name='VmwareAdminCluster', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkeonpremProjectsLocationsVmwareAdminClustersGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareAdminClusters/{vmwareAdminClustersId}:getIamPolicy', http_method='GET', method_id='gkeonprem.projects.locations.vmwareAdminClusters.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='GkeonpremProjectsLocationsVmwareAdminClustersGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists VMware admin clusters in a given project and location.

      Args:
        request: (GkeonpremProjectsLocationsVmwareAdminClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVmwareAdminClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareAdminClusters', http_method='GET', method_id='gkeonprem.projects.locations.vmwareAdminClusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['allowMissing', 'pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/vmwareAdminClusters', request_field='', request_type_name='GkeonpremProjectsLocationsVmwareAdminClustersListRequest', response_type_name='ListVmwareAdminClustersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single VMware admin cluster.

      Args:
        request: (GkeonpremProjectsLocationsVmwareAdminClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareAdminClusters/{vmwareAdminClustersId}', http_method='PATCH', method_id='gkeonprem.projects.locations.vmwareAdminClusters.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='vmwareAdminCluster', request_type_name='GkeonpremProjectsLocationsVmwareAdminClustersPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkeonpremProjectsLocationsVmwareAdminClustersSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareAdminClusters/{vmwareAdminClustersId}:setIamPolicy', http_method='POST', method_id='gkeonprem.projects.locations.vmwareAdminClusters.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkeonpremProjectsLocationsVmwareAdminClustersSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkeonpremProjectsLocationsVmwareAdminClustersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareAdminClusters/{vmwareAdminClustersId}:testIamPermissions', http_method='POST', method_id='gkeonprem.projects.locations.vmwareAdminClusters.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkeonpremProjectsLocationsVmwareAdminClustersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Unenroll(self, request, global_params=None):
        """Unenrolls an existing VMware admin cluster from the Anthos On-Prem API within a given project and location. Unenrollment removes the Cloud reference to the cluster without modifying the underlying OnPrem Resources. Clusters will continue to run; however, they will no longer be accessible through the Anthos On-Prem API or its clients.

      Args:
        request: (GkeonpremProjectsLocationsVmwareAdminClustersUnenrollRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Unenroll')
        return self._RunMethod(config, request, global_params=global_params)
    Unenroll.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareAdminClusters/{vmwareAdminClustersId}:unenroll', http_method='DELETE', method_id='gkeonprem.projects.locations.vmwareAdminClusters.unenroll', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'validateOnly'], relative_path='v1/{+name}:unenroll', request_field='', request_type_name='GkeonpremProjectsLocationsVmwareAdminClustersUnenrollRequest', response_type_name='Operation', supports_download=False)