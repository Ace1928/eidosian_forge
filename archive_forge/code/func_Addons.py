from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def Addons(self, request, global_params=None):
    """Sets the addons for a specific cluster.

      Args:
        request: (SetAddonsConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Addons')
    return self._RunMethod(config, request, global_params=global_params)