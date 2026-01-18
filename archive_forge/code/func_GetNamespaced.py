from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v3 import cloudresourcemanager_v3_messages as messages
def GetNamespaced(self, request, global_params=None):
    """Retrieves a TagValue by its namespaced name. This method will return `PERMISSION_DENIED` if the value does not exist or the user does not have permission to view it.

      Args:
        request: (CloudresourcemanagerTagValuesGetNamespacedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TagValue) The response message.
      """
    config = self.GetMethodConfig('GetNamespaced')
    return self._RunMethod(config, request, global_params=global_params)