from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetControlPlaneAccess(self, request, global_params=None):
    """Lists the service accounts with the permissions required to allow Apigee runtime-plane components access to control plane resources. Currently, the permissions required are to: 1. Allow Synchronizer to download environment data from the control plane. 2. Allow the UDCA to upload analytics data. 3. Allow the Logger component to write logs to the control plane. For more information regarding the Synchronizer, see [Configure the Synchronizer](https://cloud.google.com/apigee/docs/hybrid/latest/synchronizer-access). **Note**: Available to Apigee hybrid only.

      Args:
        request: (ApigeeOrganizationsGetControlPlaneAccessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ControlPlaneAccess) The response message.
      """
    config = self.GetMethodConfig('GetControlPlaneAccess')
    return self._RunMethod(config, request, global_params=global_params)