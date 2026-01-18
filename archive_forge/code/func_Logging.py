from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def Logging(self, request, global_params=None):
    """Sets the logging service for a specific cluster.

      Args:
        request: (SetLoggingServiceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Logging')
    return self._RunMethod(config, request, global_params=global_params)