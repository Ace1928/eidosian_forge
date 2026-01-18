from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.eventarc.v1 import eventarc_v1_messages as messages
def GetGoogleChannelConfig(self, request, global_params=None):
    """Get a GoogleChannelConfig.

      Args:
        request: (EventarcProjectsLocationsGetGoogleChannelConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleChannelConfig) The response message.
      """
    config = self.GetMethodConfig('GetGoogleChannelConfig')
    return self._RunMethod(config, request, global_params=global_params)