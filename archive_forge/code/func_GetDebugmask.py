from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetDebugmask(self, request, global_params=None):
    """Gets the debug mask singleton resource for an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetDebugmaskRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DebugMask) The response message.
      """
    config = self.GetMethodConfig('GetDebugmask')
    return self._RunMethod(config, request, global_params=global_params)