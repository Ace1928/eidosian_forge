from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.tpu.v2alpha1 import tpu_v2alpha1_messages as messages
def SimulateMaintenanceEvent(self, request, global_params=None):
    """Simulates a maintenance event.

      Args:
        request: (TpuProjectsLocationsNodesSimulateMaintenanceEventRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SimulateMaintenanceEvent')
    return self._RunMethod(config, request, global_params=global_params)