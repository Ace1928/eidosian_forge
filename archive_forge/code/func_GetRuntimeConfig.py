from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetRuntimeConfig(self, request, global_params=None):
    """Get runtime config for an organization.

      Args:
        request: (ApigeeOrganizationsGetRuntimeConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1RuntimeConfig) The response message.
      """
    config = self.GetMethodConfig('GetRuntimeConfig')
    return self._RunMethod(config, request, global_params=global_params)