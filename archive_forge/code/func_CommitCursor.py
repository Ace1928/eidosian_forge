from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
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