from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def Locations(self, request, global_params=None):
    """Sets the locations for a specific cluster. Deprecated. Use [projects.locations.clusters.update](https://cloud.google.com/kubernetes-engine/docs/reference/rest/v1/projects.locations.clusters/update) instead.

      Args:
        request: (SetLocationsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Locations')
    return self._RunMethod(config, request, global_params=global_params)