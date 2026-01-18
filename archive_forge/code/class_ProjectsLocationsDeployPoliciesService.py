from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
class ProjectsLocationsDeployPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_deployPolicies resource."""
    _NAME = 'projects_locations_deployPolicies'

    def __init__(self, client):
        super(ClouddeployV1.ProjectsLocationsDeployPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new DeployPolicy in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeployPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deployPolicies', http_method='POST', method_id='clouddeploy.projects.locations.deployPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['deployPolicyId', 'requestId', 'validateOnly'], relative_path='v1/{+parent}/deployPolicies', request_field='deployPolicy', request_type_name='ClouddeployProjectsLocationsDeployPoliciesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single DeployPolicy.

      Args:
        request: (ClouddeployProjectsLocationsDeployPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deployPolicies/{deployPoliciesId}', http_method='DELETE', method_id='clouddeploy.projects.locations.deployPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'requestId', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='ClouddeployProjectsLocationsDeployPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single DeployPolicy.

      Args:
        request: (ClouddeployProjectsLocationsDeployPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DeployPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deployPolicies/{deployPoliciesId}', http_method='GET', method_id='clouddeploy.projects.locations.deployPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ClouddeployProjectsLocationsDeployPoliciesGetRequest', response_type_name='DeployPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DeployPolicies in a given project and location.

      Args:
        request: (ClouddeployProjectsLocationsDeployPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDeployPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deployPolicies', http_method='GET', method_id='clouddeploy.projects.locations.deployPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/deployPolicies', request_field='', request_type_name='ClouddeployProjectsLocationsDeployPoliciesListRequest', response_type_name='ListDeployPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single DeployPolicy.

      Args:
        request: (ClouddeployProjectsLocationsDeployPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/deployPolicies/{deployPoliciesId}', http_method='PATCH', method_id='clouddeploy.projects.locations.deployPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='deployPolicy', request_type_name='ClouddeployProjectsLocationsDeployPoliciesPatchRequest', response_type_name='Operation', supports_download=False)