from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
def UpdateSigned(self, request, global_params=None):
    """Updates a signed device.

      Args:
        request: (SasportalNodesDevicesUpdateSignedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
    config = self.GetMethodConfig('UpdateSigned')
    return self._RunMethod(config, request, global_params=global_params)