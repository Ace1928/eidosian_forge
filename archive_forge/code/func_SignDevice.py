from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
def SignDevice(self, request, global_params=None):
    """Signs a device.

      Args:
        request: (SasportalNodesDevicesSignDeviceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalEmpty) The response message.
      """
    config = self.GetMethodConfig('SignDevice')
    return self._RunMethod(config, request, global_params=global_params)