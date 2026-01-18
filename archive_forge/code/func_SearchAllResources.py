from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def SearchAllResources(self, request, global_params=None):
    """Searches all Google Cloud resources within the specified scope, such as a project, folder, or organization. The caller must be granted the `cloudasset.assets.searchAllResources` permission on the desired scope, otherwise the request will be rejected.

      Args:
        request: (CloudassetSearchAllResourcesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchAllResourcesResponse) The response message.
      """
    config = self.GetMethodConfig('SearchAllResources')
    return self._RunMethod(config, request, global_params=global_params)