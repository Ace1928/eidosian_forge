from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
class ProjectsLocationsOrchestrationClustersDeploymentsService(base_api.BaseApiService):
    """Service class for the projects_locations_orchestrationClusters_deployments resource."""
    _NAME = 'projects_locations_orchestrationClusters_deployments'

    def __init__(self, client):
        super(TelcoautomationV1.ProjectsLocationsOrchestrationClustersDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Apply(self, request, global_params=None):
        """Applies the deployment's YAML files to the parent orchestration cluster.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsApplyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Deployment) The response message.
      """
        config = self.GetMethodConfig('Apply')
        return self._RunMethod(config, request, global_params=global_params)
    Apply.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}:apply', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.apply', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:apply', request_field='applyDeploymentRequest', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsApplyRequest', response_type_name='Deployment', supports_download=False)

    def ComputeDeploymentStatus(self, request, global_params=None):
        """Returns the requested deployment status.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsComputeDeploymentStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeDeploymentStatusResponse) The response message.
      """
        config = self.GetMethodConfig('ComputeDeploymentStatus')
        return self._RunMethod(config, request, global_params=global_params)
    ComputeDeploymentStatus.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}:computeDeploymentStatus', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.computeDeploymentStatus', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:computeDeploymentStatus', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsComputeDeploymentStatusRequest', response_type_name='ComputeDeploymentStatusResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a deployment.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Deployment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.create', ordered_params=['parent'], path_params=['parent'], query_params=['deploymentId'], relative_path='v1/{+parent}/deployments', request_field='deployment', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsCreateRequest', response_type_name='Deployment', supports_download=False)

    def Discard(self, request, global_params=None):
        """Discards the changes in a deployment and reverts the deployment to the last approved deployment revision. No changes take place if a deployment does not have revisions.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsDiscardRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiscardDeploymentChangesResponse) The response message.
      """
        config = self.GetMethodConfig('Discard')
        return self._RunMethod(config, request, global_params=global_params)
    Discard.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}:discard', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.discard', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:discard', request_field='discardDeploymentChangesRequest', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsDiscardRequest', response_type_name='DiscardDeploymentChangesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested deployment.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Deployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsGetRequest', response_type_name='Deployment', supports_download=False)

    def List(self, request, global_params=None):
        """List all deployments.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/deployments', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsListRequest', response_type_name='ListDeploymentsResponse', supports_download=False)

    def ListRevisions(self, request, global_params=None):
        """List deployment revisions of a given deployment.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsListRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDeploymentRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('ListRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    ListRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}:listRevisions', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.listRevisions', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+name}:listRevisions', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsListRevisionsRequest', response_type_name='ListDeploymentRevisionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a deployment.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Deployment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}', http_method='PATCH', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='deployment', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsPatchRequest', response_type_name='Deployment', supports_download=False)

    def Remove(self, request, global_params=None):
        """Removes the deployment by marking it as DELETING. Post which deployment and it's revisions gets deleted.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsRemoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Remove')
        return self._RunMethod(config, request, global_params=global_params)
    Remove.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}:remove', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.remove', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:remove', request_field='removeDeploymentRequest', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsRemoveRequest', response_type_name='Empty', supports_download=False)

    def Rollback(self, request, global_params=None):
        """Rollback the active deployment to the given past approved deployment revision.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsRollbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Deployment) The response message.
      """
        config = self.GetMethodConfig('Rollback')
        return self._RunMethod(config, request, global_params=global_params)
    Rollback.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments/{deploymentsId}:rollback', http_method='POST', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.rollback', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rollback', request_field='rollbackDeploymentRequest', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsRollbackRequest', response_type_name='Deployment', supports_download=False)

    def SearchRevisions(self, request, global_params=None):
        """Searches across deployment revisions.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsSearchRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchDeploymentRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    SearchRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/orchestrationClusters/{orchestrationClustersId}/deployments:searchRevisions', http_method='GET', method_id='telcoautomation.projects.locations.orchestrationClusters.deployments.searchRevisions', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'query'], relative_path='v1/{+parent}/deployments:searchRevisions', request_field='', request_type_name='TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsSearchRevisionsRequest', response_type_name='SearchDeploymentRevisionsResponse', supports_download=False)