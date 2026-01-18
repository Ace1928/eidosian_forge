from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v1 import cloudresourcemanager_v1_messages as messages
def ClearOrgPolicy(self, request, global_params=None):
    """Clears a `Policy` from a resource.

      Args:
        request: (CloudresourcemanagerProjectsClearOrgPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('ClearOrgPolicy')
    return self._RunMethod(config, request, global_params=global_params)