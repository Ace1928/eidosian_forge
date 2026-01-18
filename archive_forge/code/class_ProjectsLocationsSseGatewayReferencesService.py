from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
class ProjectsLocationsSseGatewayReferencesService(base_api.BaseApiService):
    """Service class for the projects_locations_sseGatewayReferences resource."""
    _NAME = 'projects_locations_sseGatewayReferences'

    def __init__(self, client):
        super(NetworksecurityV1alpha1.ProjectsLocationsSseGatewayReferencesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single SSEGatewayReference.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewayReferencesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SSEGatewayReference) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/sseGatewayReferences/{sseGatewayReferencesId}', http_method='GET', method_id='networksecurity.projects.locations.sseGatewayReferences.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsSseGatewayReferencesGetRequest', response_type_name='SSEGatewayReference', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SSEGatewayReferences in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewayReferencesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSSEGatewayReferencesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/sseGatewayReferences', http_method='GET', method_id='networksecurity.projects.locations.sseGatewayReferences.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/sseGatewayReferences', request_field='', request_type_name='NetworksecurityProjectsLocationsSseGatewayReferencesListRequest', response_type_name='ListSSEGatewayReferencesResponse', supports_download=False)