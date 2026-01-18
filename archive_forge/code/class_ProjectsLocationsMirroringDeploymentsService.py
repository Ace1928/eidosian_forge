from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
class ProjectsLocationsMirroringDeploymentsService(base_api.BaseApiService):
    """Service class for the projects_locations_mirroringDeployments resource."""
    _NAME = 'projects_locations_mirroringDeployments'

    def __init__(self, client):
        super(NetworksecurityV1alpha1.ProjectsLocationsMirroringDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new MirroringDeployment in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringDeployments', http_method='POST', method_id='networksecurity.projects.locations.mirroringDeployments.create', ordered_params=['parent'], path_params=['parent'], query_params=['mirroringDeploymentId', 'requestId'], relative_path='v1alpha1/{+parent}/mirroringDeployments', request_field='mirroringDeployment', request_type_name='NetworksecurityProjectsLocationsMirroringDeploymentsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single MirroringDeployment.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringDeploymentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringDeployments/{mirroringDeploymentsId}', http_method='DELETE', method_id='networksecurity.projects.locations.mirroringDeployments.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsMirroringDeploymentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single MirroringDeployment.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MirroringDeployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringDeployments/{mirroringDeploymentsId}', http_method='GET', method_id='networksecurity.projects.locations.mirroringDeployments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsMirroringDeploymentsGetRequest', response_type_name='MirroringDeployment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists MirroringDeployments in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMirroringDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringDeployments', http_method='GET', method_id='networksecurity.projects.locations.mirroringDeployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/mirroringDeployments', request_field='', request_type_name='NetworksecurityProjectsLocationsMirroringDeploymentsListRequest', response_type_name='ListMirroringDeploymentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a single MirroringDeployment.

      Args:
        request: (NetworksecurityProjectsLocationsMirroringDeploymentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/mirroringDeployments/{mirroringDeploymentsId}', http_method='PATCH', method_id='networksecurity.projects.locations.mirroringDeployments.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='mirroringDeployment', request_type_name='NetworksecurityProjectsLocationsMirroringDeploymentsPatchRequest', response_type_name='Operation', supports_download=False)