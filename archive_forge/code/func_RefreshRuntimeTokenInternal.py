from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v1 import notebooks_v1_messages as messages
def RefreshRuntimeTokenInternal(self, request, global_params=None):
    """Gets an access token for the consumer service account that the customer attached to the runtime. Only accessible from the tenant instance.

      Args:
        request: (NotebooksProjectsLocationsRuntimesRefreshRuntimeTokenInternalRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RefreshRuntimeTokenInternalResponse) The response message.
      """
    config = self.GetMethodConfig('RefreshRuntimeTokenInternal')
    return self._RunMethod(config, request, global_params=global_params)