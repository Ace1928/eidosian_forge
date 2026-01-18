from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class CursorProjectsLocationsSubscriptionsService(base_api.BaseApiService):
    """Service class for the cursor_projects_locations_subscriptions resource."""
    _NAME = 'cursor_projects_locations_subscriptions'

    def __init__(self, client):
        super(PubsubliteV1.CursorProjectsLocationsSubscriptionsService, self).__init__(client)
        self._upload_configs = {}

    def CommitCursor(self, request, global_params=None):
        """Updates the committed cursor.

      Args:
        request: (PubsubliteCursorProjectsLocationsSubscriptionsCommitCursorRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CommitCursorResponse) The response message.
      """
        config = self.GetMethodConfig('CommitCursor')
        return self._RunMethod(config, request, global_params=global_params)
    CommitCursor.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/cursor/projects/{projectsId}/locations/{locationsId}/subscriptions/{subscriptionsId}:commitCursor', http_method='POST', method_id='pubsublite.cursor.projects.locations.subscriptions.commitCursor', ordered_params=['subscription'], path_params=['subscription'], query_params=[], relative_path='v1/cursor/{+subscription}:commitCursor', request_field='commitCursorRequest', request_type_name='PubsubliteCursorProjectsLocationsSubscriptionsCommitCursorRequest', response_type_name='CommitCursorResponse', supports_download=False)