from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as messages
class ProjectsLocationsApplicationsDeploymentsService(base_api.BaseApiService):
    """Service class for the projects_locations_applications_deployments resource."""
    _NAME = 'projects_locations_applications_deployments'

    def __init__(self, client):
        super(RunappsV1alpha1.ProjectsLocationsApplicationsDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Deployment in a given project and location.

      Args:
        request: (RunappsProjectsLocationsApplicationsDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/deployments', http_method='POST', method_id='runapps.projects.locations.applications.deployments.create', ordered_params=['parent'], path_params=['parent'], query_params=['deploymentId', 'requestId', 'validateOnly'], relative_path='v1alpha1/{+parent}/deployments', request_field='deployment', request_type_name='RunappsProjectsLocationsApplicationsDeploymentsCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Deployment.

      Args:
        request: (RunappsProjectsLocationsApplicationsDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Deployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/deployments/{deploymentsId}', http_method='GET', method_id='runapps.projects.locations.applications.deployments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='RunappsProjectsLocationsApplicationsDeploymentsGetRequest', response_type_name='Deployment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Deployments in a given project and location.

      Args:
        request: (RunappsProjectsLocationsApplicationsDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/applications/{applicationsId}/deployments', http_method='GET', method_id='runapps.projects.locations.applications.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/deployments', request_field='', request_type_name='RunappsProjectsLocationsApplicationsDeploymentsListRequest', response_type_name='ListDeploymentsResponse', supports_download=False)