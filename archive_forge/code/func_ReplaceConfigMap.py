from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
def ReplaceConfigMap(self, request, global_params=None):
    """Rpc to replace a ConfigMap.

      Args:
        request: (AnthoseventsProjectsLocationsNamespacesConfigmapsReplaceConfigMapRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigMap) The response message.
      """
    config = self.GetMethodConfig('ReplaceConfigMap')
    return self._RunMethod(config, request, global_params=global_params)