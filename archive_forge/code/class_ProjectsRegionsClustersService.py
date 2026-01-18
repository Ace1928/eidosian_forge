from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsRegionsClustersService(base_api.BaseApiService):
    """Service class for the projects_regions_clusters resource."""
    _NAME = 'projects_regions_clusters'

    def __init__(self, client):
        super(DataprocV1.ProjectsRegionsClustersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a cluster in a project. The returned Operation.metadata will be ClusterOperationMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#clusteroperationmetadata).

      Args:
        request: (DataprocProjectsRegionsClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataproc.projects.regions.clusters.create', ordered_params=['projectId', 'region'], path_params=['projectId', 'region'], query_params=['actionOnFailedPrimaryWorkers', 'requestId'], relative_path='v1/projects/{projectId}/regions/{region}/clusters', request_field='cluster', request_type_name='DataprocProjectsRegionsClustersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a cluster in a project. The returned Operation.metadata will be ClusterOperationMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#clusteroperationmetadata).

      Args:
        request: (DataprocProjectsRegionsClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='dataproc.projects.regions.clusters.delete', ordered_params=['projectId', 'region', 'clusterName'], path_params=['clusterName', 'projectId', 'region'], query_params=['clusterUuid', 'gracefulTerminationTimeout', 'requestId'], relative_path='v1/projects/{projectId}/regions/{region}/clusters/{clusterName}', request_field='', request_type_name='DataprocProjectsRegionsClustersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Diagnose(self, request, global_params=None):
        """Gets cluster diagnostic information. The returned Operation.metadata will be ClusterOperationMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#clusteroperationmetadata). After the operation completes, Operation.response contains DiagnoseClusterResults (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#diagnoseclusterresults).

      Args:
        request: (DataprocProjectsRegionsClustersDiagnoseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Diagnose')
        return self._RunMethod(config, request, global_params=global_params)
    Diagnose.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataproc.projects.regions.clusters.diagnose', ordered_params=['projectId', 'region', 'clusterName'], path_params=['clusterName', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/clusters/{clusterName}:diagnose', request_field='diagnoseClusterRequest', request_type_name='DataprocProjectsRegionsClustersDiagnoseRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the resource representation for a cluster in a project.

      Args:
        request: (DataprocProjectsRegionsClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Cluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataproc.projects.regions.clusters.get', ordered_params=['projectId', 'region', 'clusterName'], path_params=['clusterName', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/clusters/{clusterName}', request_field='', request_type_name='DataprocProjectsRegionsClustersGetRequest', response_type_name='Cluster', supports_download=False)

    def GetClusterAsTemplate(self, request, global_params=None):
        """Exports a template for a cluster in a project that can be used in future CreateCluster requests.

      Args:
        request: (DataprocProjectsRegionsClustersGetClusterAsTemplateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Cluster) The response message.
      """
        config = self.GetMethodConfig('GetClusterAsTemplate')
        return self._RunMethod(config, request, global_params=global_params)
    GetClusterAsTemplate.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataproc.projects.regions.clusters.getClusterAsTemplate', ordered_params=['projectId', 'region', 'clusterName'], path_params=['clusterName', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/clusters/{clusterName}:getClusterAsTemplate', request_field='', request_type_name='DataprocProjectsRegionsClustersGetClusterAsTemplateRequest', response_type_name='Cluster', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DataprocProjectsRegionsClustersGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}:getIamPolicy', http_method='POST', method_id='dataproc.projects.regions.clusters.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='DataprocProjectsRegionsClustersGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def InjectCredentials(self, request, global_params=None):
        """Inject encrypted credentials into all of the VMs in a cluster.The target cluster must be a personal auth cluster assigned to the user who is issuing the RPC.

      Args:
        request: (DataprocProjectsRegionsClustersInjectCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('InjectCredentials')
        return self._RunMethod(config, request, global_params=global_params)
    InjectCredentials.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}:injectCredentials', http_method='POST', method_id='dataproc.projects.regions.clusters.injectCredentials', ordered_params=['project', 'region', 'cluster'], path_params=['cluster', 'project', 'region'], query_params=[], relative_path='v1/{+project}/{+region}/{+cluster}:injectCredentials', request_field='injectCredentialsRequest', request_type_name='DataprocProjectsRegionsClustersInjectCredentialsRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all regions/{region}/clusters in a project alphabetically.

      Args:
        request: (DataprocProjectsRegionsClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataproc.projects.regions.clusters.list', ordered_params=['projectId', 'region'], path_params=['projectId', 'region'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/projects/{projectId}/regions/{region}/clusters', request_field='', request_type_name='DataprocProjectsRegionsClustersListRequest', response_type_name='ListClustersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a cluster in a project. The returned Operation.metadata will be ClusterOperationMetadata (https://cloud.google.com/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#clusteroperationmetadata). The cluster must be in a RUNNING state or an error is returned.

      Args:
        request: (DataprocProjectsRegionsClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='dataproc.projects.regions.clusters.patch', ordered_params=['projectId', 'region', 'clusterName'], path_params=['clusterName', 'projectId', 'region'], query_params=['gracefulDecommissionTimeout', 'requestId', 'updateMask'], relative_path='v1/projects/{projectId}/regions/{region}/clusters/{clusterName}', request_field='cluster', request_type_name='DataprocProjectsRegionsClustersPatchRequest', response_type_name='Operation', supports_download=False)

    def Repair(self, request, global_params=None):
        """Repairs a cluster.

      Args:
        request: (DataprocProjectsRegionsClustersRepairRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Repair')
        return self._RunMethod(config, request, global_params=global_params)
    Repair.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataproc.projects.regions.clusters.repair', ordered_params=['projectId', 'region', 'clusterName'], path_params=['clusterName', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/clusters/{clusterName}:repair', request_field='repairClusterRequest', request_type_name='DataprocProjectsRegionsClustersRepairRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (DataprocProjectsRegionsClustersSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}:setIamPolicy', http_method='POST', method_id='dataproc.projects.regions.clusters.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DataprocProjectsRegionsClustersSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Start(self, request, global_params=None):
        """Starts a cluster in a project.

      Args:
        request: (DataprocProjectsRegionsClustersStartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Start')
        return self._RunMethod(config, request, global_params=global_params)
    Start.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataproc.projects.regions.clusters.start', ordered_params=['projectId', 'region', 'clusterName'], path_params=['clusterName', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/clusters/{clusterName}:start', request_field='startClusterRequest', request_type_name='DataprocProjectsRegionsClustersStartRequest', response_type_name='Operation', supports_download=False)

    def Stop(self, request, global_params=None):
        """Stops a cluster in a project.

      Args:
        request: (DataprocProjectsRegionsClustersStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataproc.projects.regions.clusters.stop', ordered_params=['projectId', 'region', 'clusterName'], path_params=['clusterName', 'projectId', 'region'], query_params=[], relative_path='v1/projects/{projectId}/regions/{region}/clusters/{clusterName}:stop', request_field='stopClusterRequest', request_type_name='DataprocProjectsRegionsClustersStopRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DataprocProjectsRegionsClustersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/regions/{regionsId}/clusters/{clustersId}:testIamPermissions', http_method='POST', method_id='dataproc.projects.regions.clusters.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DataprocProjectsRegionsClustersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)