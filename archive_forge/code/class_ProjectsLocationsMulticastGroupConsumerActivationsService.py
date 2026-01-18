from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsMulticastGroupConsumerActivationsService(base_api.BaseApiService):
    """Service class for the projects_locations_multicastGroupConsumerActivations resource."""
    _NAME = 'projects_locations_multicastGroupConsumerActivations'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsMulticastGroupConsumerActivationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new multicast group consumer activation in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupConsumerActivations', http_method='POST', method_id='networkservices.projects.locations.multicastGroupConsumerActivations.create', ordered_params=['parent'], path_params=['parent'], query_params=['multicastGroupConsumerActivationId', 'requestId'], relative_path='v1/{+parent}/multicastGroupConsumerActivations', request_field='multicastGroupConsumerActivation', request_type_name='NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single multicast group consumer activation.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupConsumerActivations/{multicastGroupConsumerActivationsId}', http_method='DELETE', method_id='networkservices.projects.locations.multicastGroupConsumerActivations.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single multicast group consumer activation.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MulticastGroupConsumerActivation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupConsumerActivations/{multicastGroupConsumerActivationsId}', http_method='GET', method_id='networkservices.projects.locations.multicastGroupConsumerActivations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsGetRequest', response_type_name='MulticastGroupConsumerActivation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists multicast group consumer activations in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMulticastGroupConsumerActivationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupConsumerActivations', http_method='GET', method_id='networkservices.projects.locations.multicastGroupConsumerActivations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/multicastGroupConsumerActivations', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsListRequest', response_type_name='ListMulticastGroupConsumerActivationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single multicast group consumer activation.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroupConsumerActivations/{multicastGroupConsumerActivationsId}', http_method='PATCH', method_id='networkservices.projects.locations.multicastGroupConsumerActivations.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='multicastGroupConsumerActivation', request_type_name='NetworkservicesProjectsLocationsMulticastGroupConsumerActivationsPatchRequest', response_type_name='Operation', supports_download=False)