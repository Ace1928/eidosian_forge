from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class AdminProjectsLocationsTopicsSubscriptionsService(base_api.BaseApiService):
    """Service class for the admin_projects_locations_topics_subscriptions resource."""
    _NAME = 'admin_projects_locations_topics_subscriptions'

    def __init__(self, client):
        super(PubsubliteV1.AdminProjectsLocationsTopicsSubscriptionsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the subscriptions attached to the specified topic.

      Args:
        request: (PubsubliteAdminProjectsLocationsTopicsSubscriptionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTopicSubscriptionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/topics/{topicsId}/subscriptions', http_method='GET', method_id='pubsublite.admin.projects.locations.topics.subscriptions.list', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v1/admin/{+name}/subscriptions', request_field='', request_type_name='PubsubliteAdminProjectsLocationsTopicsSubscriptionsListRequest', response_type_name='ListTopicSubscriptionsResponse', supports_download=False)