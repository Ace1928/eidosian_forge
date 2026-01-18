from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
class ProjectsLocationsSseGatewaysService(base_api.BaseApiService):
    """Service class for the projects_locations_sseGateways resource."""
    _NAME = 'projects_locations_sseGateways'

    def __init__(self, client):
        super(NetworksecurityV1alpha1.ProjectsLocationsSseGatewaysService, self).__init__(client)
        self._upload_configs = {}

    def AttachAppNetwork(self, request, global_params=None):
        """Attaches an app network to a SSEGateway.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewaysAttachAppNetworkRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AttachAppNetwork')
        return self._RunMethod(config, request, global_params=global_params)
    AttachAppNetwork.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/sseGateways/{sseGatewaysId}:attachAppNetwork', http_method='POST', method_id='networksecurity.projects.locations.sseGateways.attachAppNetwork', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:attachAppNetwork', request_field='attachAppNetworkRequest', request_type_name='NetworksecurityProjectsLocationsSseGatewaysAttachAppNetworkRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new SSEGateway in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewaysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/sseGateways', http_method='POST', method_id='networksecurity.projects.locations.sseGateways.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'sseGatewayId'], relative_path='v1alpha1/{+parent}/sseGateways', request_field='sSEGateway', request_type_name='NetworksecurityProjectsLocationsSseGatewaysCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single SSEGateway.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewaysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/sseGateways/{sseGatewaysId}', http_method='DELETE', method_id='networksecurity.projects.locations.sseGateways.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsSseGatewaysDeleteRequest', response_type_name='Operation', supports_download=False)

    def DetachAppNetwork(self, request, global_params=None):
        """Detaches an app network from a SSEGateway.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewaysDetachAppNetworkRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DetachAppNetwork')
        return self._RunMethod(config, request, global_params=global_params)
    DetachAppNetwork.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/sseGateways/{sseGatewaysId}:detachAppNetwork', http_method='POST', method_id='networksecurity.projects.locations.sseGateways.detachAppNetwork', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:detachAppNetwork', request_field='detachAppNetworkRequest', request_type_name='NetworksecurityProjectsLocationsSseGatewaysDetachAppNetworkRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single SSEGateway.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewaysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SSEGateway) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/sseGateways/{sseGatewaysId}', http_method='GET', method_id='networksecurity.projects.locations.sseGateways.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsSseGatewaysGetRequest', response_type_name='SSEGateway', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SSEGateways in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsSseGatewaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSSEGatewaysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/sseGateways', http_method='GET', method_id='networksecurity.projects.locations.sseGateways.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/sseGateways', request_field='', request_type_name='NetworksecurityProjectsLocationsSseGatewaysListRequest', response_type_name='ListSSEGatewaysResponse', supports_download=False)