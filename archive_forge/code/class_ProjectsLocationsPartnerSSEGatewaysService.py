from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
class ProjectsLocationsPartnerSSEGatewaysService(base_api.BaseApiService):
    """Service class for the projects_locations_partnerSSEGateways resource."""
    _NAME = 'projects_locations_partnerSSEGateways'

    def __init__(self, client):
        super(NetworksecurityV1alpha1.ProjectsLocationsPartnerSSEGatewaysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new PartnerSSEGateway in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSEGatewaysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSEGateways', http_method='POST', method_id='networksecurity.projects.locations.partnerSSEGateways.create', ordered_params=['parent'], path_params=['parent'], query_params=['partnerSseGatewayId', 'requestId'], relative_path='v1alpha1/{+parent}/partnerSSEGateways', request_field='partnerSSEGateway', request_type_name='NetworksecurityProjectsLocationsPartnerSSEGatewaysCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single PartnerSSEGateway.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSEGatewaysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSEGateways/{partnerSSEGatewaysId}', http_method='DELETE', method_id='networksecurity.projects.locations.partnerSSEGateways.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsPartnerSSEGatewaysDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single PartnerSSEGateway.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSEGatewaysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartnerSSEGateway) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSEGateways/{partnerSSEGatewaysId}', http_method='GET', method_id='networksecurity.projects.locations.partnerSSEGateways.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsPartnerSSEGatewaysGetRequest', response_type_name='PartnerSSEGateway', supports_download=False)

    def List(self, request, global_params=None):
        """Lists PartnerSSEGateways in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSEGatewaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPartnerSSEGatewaysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSEGateways', http_method='GET', method_id='networksecurity.projects.locations.partnerSSEGateways.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/partnerSSEGateways', request_field='', request_type_name='NetworksecurityProjectsLocationsPartnerSSEGatewaysListRequest', response_type_name='ListPartnerSSEGatewaysResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a single PartnerSSEGateway.

      Args:
        request: (NetworksecurityProjectsLocationsPartnerSSEGatewaysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/partnerSSEGateways/{partnerSSEGatewaysId}', http_method='PATCH', method_id='networksecurity.projects.locations.partnerSSEGateways.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='partnerSSEGateway', request_type_name='NetworksecurityProjectsLocationsPartnerSSEGatewaysPatchRequest', response_type_name='Operation', supports_download=False)