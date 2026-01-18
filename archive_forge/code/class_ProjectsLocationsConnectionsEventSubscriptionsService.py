from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectors.v1 import connectors_v1_messages as messages
class ProjectsLocationsConnectionsEventSubscriptionsService(base_api.BaseApiService):
    """Service class for the projects_locations_connections_eventSubscriptions resource."""
    _NAME = 'projects_locations_connections_eventSubscriptions'

    def __init__(self, client):
        super(ConnectorsV1.ProjectsLocationsConnectionsEventSubscriptionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new EventSubscription in a given project,location and connection.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsEventSubscriptionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}/eventSubscriptions', http_method='POST', method_id='connectors.projects.locations.connections.eventSubscriptions.create', ordered_params=['parent'], path_params=['parent'], query_params=['eventSubscriptionId'], relative_path='v1/{+parent}/eventSubscriptions', request_field='eventSubscription', request_type_name='ConnectorsProjectsLocationsConnectionsEventSubscriptionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single EventSubscription.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsEventSubscriptionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}/eventSubscriptions/{eventSubscriptionsId}', http_method='DELETE', method_id='connectors.projects.locations.connections.eventSubscriptions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ConnectorsProjectsLocationsConnectionsEventSubscriptionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single EventSubscription.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsEventSubscriptionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EventSubscription) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}/eventSubscriptions/{eventSubscriptionsId}', http_method='GET', method_id='connectors.projects.locations.connections.eventSubscriptions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ConnectorsProjectsLocationsConnectionsEventSubscriptionsGetRequest', response_type_name='EventSubscription', supports_download=False)

    def List(self, request, global_params=None):
        """List EventSubscriptions in a given project,location and connection.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsEventSubscriptionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEventSubscriptionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}/eventSubscriptions', http_method='GET', method_id='connectors.projects.locations.connections.eventSubscriptions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/eventSubscriptions', request_field='', request_type_name='ConnectorsProjectsLocationsConnectionsEventSubscriptionsListRequest', response_type_name='ListEventSubscriptionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single EventSubscription.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsEventSubscriptionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}/eventSubscriptions/{eventSubscriptionsId}', http_method='PATCH', method_id='connectors.projects.locations.connections.eventSubscriptions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='eventSubscription', request_type_name='ConnectorsProjectsLocationsConnectionsEventSubscriptionsPatchRequest', response_type_name='Operation', supports_download=False)

    def Retry(self, request, global_params=None):
        """RetryEventSubscription retries the registration of Subscription.

      Args:
        request: (ConnectorsProjectsLocationsConnectionsEventSubscriptionsRetryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Retry')
        return self._RunMethod(config, request, global_params=global_params)
    Retry.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/connections/{connectionsId}/eventSubscriptions/{eventSubscriptionsId}:retry', http_method='POST', method_id='connectors.projects.locations.connections.eventSubscriptions.retry', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:retry', request_field='retryEventSubscriptionRequest', request_type_name='ConnectorsProjectsLocationsConnectionsEventSubscriptionsRetryRequest', response_type_name='Operation', supports_download=False)