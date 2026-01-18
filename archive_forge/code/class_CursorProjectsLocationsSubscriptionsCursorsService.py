from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class CursorProjectsLocationsSubscriptionsCursorsService(base_api.BaseApiService):
    """Service class for the cursor_projects_locations_subscriptions_cursors resource."""
    _NAME = 'cursor_projects_locations_subscriptions_cursors'

    def __init__(self, client):
        super(PubsubliteV1.CursorProjectsLocationsSubscriptionsCursorsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Returns all committed cursor information for a subscription.

      Args:
        request: (PubsubliteCursorProjectsLocationsSubscriptionsCursorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPartitionCursorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/cursor/projects/{projectsId}/locations/{locationsId}/subscriptions/{subscriptionsId}/cursors', http_method='GET', method_id='pubsublite.cursor.projects.locations.subscriptions.cursors.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/cursor/{+parent}/cursors', request_field='', request_type_name='PubsubliteCursorProjectsLocationsSubscriptionsCursorsListRequest', response_type_name='ListPartitionCursorsResponse', supports_download=False)