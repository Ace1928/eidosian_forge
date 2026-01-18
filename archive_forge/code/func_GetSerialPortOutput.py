from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetSerialPortOutput(self, request, global_params=None):
    """Returns the last 1 MB of serial port output from the specified instance.

      Args:
        request: (ComputeInstancesGetSerialPortOutputRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SerialPortOutput) The response message.
      """
    config = self.GetMethodConfig('GetSerialPortOutput')
    return self._RunMethod(config, request, global_params=global_params)