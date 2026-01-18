from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1beta1 import networkservices_v1beta1_messages as messages
class ProjectsLocationsMessageSubscriptionsService(base_api.BaseApiService):
    """Service class for the projects_locations_messageSubscriptions resource."""
    _NAME = 'projects_locations_messageSubscriptions'

    def __init__(self, client):
        super(NetworkservicesV1beta1.ProjectsLocationsMessageSubscriptionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new MessageSubscription in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMessageSubscriptionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messageSubscriptions', http_method='POST', method_id='networkservices.projects.locations.messageSubscriptions.create', ordered_params=['parent'], path_params=['parent'], query_params=['messageSubscriptionId'], relative_path='v1beta1/{+parent}/messageSubscriptions', request_field='messageSubscription', request_type_name='NetworkservicesProjectsLocationsMessageSubscriptionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single MessageSubscription.

      Args:
        request: (NetworkservicesProjectsLocationsMessageSubscriptionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messageSubscriptions/{messageSubscriptionsId}', http_method='DELETE', method_id='networkservices.projects.locations.messageSubscriptions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMessageSubscriptionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single MessageSubscription.

      Args:
        request: (NetworkservicesProjectsLocationsMessageSubscriptionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MessageSubscription) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messageSubscriptions/{messageSubscriptionsId}', http_method='GET', method_id='networkservices.projects.locations.messageSubscriptions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMessageSubscriptionsGetRequest', response_type_name='MessageSubscription', supports_download=False)

    def List(self, request, global_params=None):
        """Lists MessageSubscription in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMessageSubscriptionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMessageSubscriptionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messageSubscriptions', http_method='GET', method_id='networkservices.projects.locations.messageSubscriptions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/messageSubscriptions', request_field='', request_type_name='NetworkservicesProjectsLocationsMessageSubscriptionsListRequest', response_type_name='ListMessageSubscriptionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single MessageSubscription.

      Args:
        request: (NetworkservicesProjectsLocationsMessageSubscriptionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/messageSubscriptions/{messageSubscriptionsId}', http_method='PATCH', method_id='networkservices.projects.locations.messageSubscriptions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta1/{+name}', request_field='messageSubscription', request_type_name='NetworkservicesProjectsLocationsMessageSubscriptionsPatchRequest', response_type_name='Operation', supports_download=False)