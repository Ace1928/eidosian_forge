from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class AdminProjectsLocationsSubscriptionsService(base_api.BaseApiService):
    """Service class for the admin_projects_locations_subscriptions resource."""
    _NAME = 'admin_projects_locations_subscriptions'

    def __init__(self, client):
        super(PubsubliteV1.AdminProjectsLocationsSubscriptionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new subscription.

      Args:
        request: (PubsubliteAdminProjectsLocationsSubscriptionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Subscription) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/subscriptions', http_method='POST', method_id='pubsublite.admin.projects.locations.subscriptions.create', ordered_params=['parent'], path_params=['parent'], query_params=['skipBacklog', 'subscriptionId'], relative_path='v1/admin/{+parent}/subscriptions', request_field='subscription', request_type_name='PubsubliteAdminProjectsLocationsSubscriptionsCreateRequest', response_type_name='Subscription', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified subscription.

      Args:
        request: (PubsubliteAdminProjectsLocationsSubscriptionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/subscriptions/{subscriptionsId}', http_method='DELETE', method_id='pubsublite.admin.projects.locations.subscriptions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/admin/{+name}', request_field='', request_type_name='PubsubliteAdminProjectsLocationsSubscriptionsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the subscription configuration.

      Args:
        request: (PubsubliteAdminProjectsLocationsSubscriptionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Subscription) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/subscriptions/{subscriptionsId}', http_method='GET', method_id='pubsublite.admin.projects.locations.subscriptions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/admin/{+name}', request_field='', request_type_name='PubsubliteAdminProjectsLocationsSubscriptionsGetRequest', response_type_name='Subscription', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of subscriptions for the given project.

      Args:
        request: (PubsubliteAdminProjectsLocationsSubscriptionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSubscriptionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/subscriptions', http_method='GET', method_id='pubsublite.admin.projects.locations.subscriptions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/admin/{+parent}/subscriptions', request_field='', request_type_name='PubsubliteAdminProjectsLocationsSubscriptionsListRequest', response_type_name='ListSubscriptionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates properties of the specified subscription.

      Args:
        request: (PubsubliteAdminProjectsLocationsSubscriptionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Subscription) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/subscriptions/{subscriptionsId}', http_method='PATCH', method_id='pubsublite.admin.projects.locations.subscriptions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/admin/{+name}', request_field='subscription', request_type_name='PubsubliteAdminProjectsLocationsSubscriptionsPatchRequest', response_type_name='Subscription', supports_download=False)

    def Seek(self, request, global_params=None):
        """Performs an out-of-band seek for a subscription to a specified target, which may be timestamps or named positions within the message backlog. Seek translates these targets to cursors for each partition and orchestrates subscribers to start consuming messages from these seek cursors. If an operation is returned, the seek has been registered and subscribers will eventually receive messages from the seek cursors (i.e. eventual consistency), as long as they are using a minimum supported client library version and not a system that tracks cursors independently of Pub/Sub Lite (e.g. Apache Beam, Dataflow, Spark). The seek operation will fail for unsupported clients. If clients would like to know when subscribers react to the seek (or not), they can poll the operation. The seek operation will succeed and complete once subscribers are ready to receive messages from the seek cursors for all partitions of the topic. This means that the seek operation will not complete until all subscribers come online. If the previous seek operation has not yet completed, it will be aborted and the new invocation of seek will supersede it.

      Args:
        request: (PubsubliteAdminProjectsLocationsSubscriptionsSeekRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Seek')
        return self._RunMethod(config, request, global_params=global_params)
    Seek.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/subscriptions/{subscriptionsId}:seek', http_method='POST', method_id='pubsublite.admin.projects.locations.subscriptions.seek', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/admin/{+name}:seek', request_field='seekSubscriptionRequest', request_type_name='PubsubliteAdminProjectsLocationsSubscriptionsSeekRequest', response_type_name='Operation', supports_download=False)