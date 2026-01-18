from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def QueryVersionConfig(self, request, global_params=None):
    """Queries the VMware user cluster version config.

      Args:
        request: (GkeonpremProjectsLocationsVmwareClustersQueryVersionConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryVmwareVersionConfigResponse) The response message.
      """
    config = self.GetMethodConfig('QueryVersionConfig')
    return self._RunMethod(config, request, global_params=global_params)