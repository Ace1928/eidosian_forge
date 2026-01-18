from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def SetSyncAuthorization(self, request, global_params=None):
    """Sets the permissions required to allow the Synchronizer to download environment data from the control plane. You must call this API to enable proper functioning of hybrid. Pass the ETag when calling `setSyncAuthorization` to ensure that you are updating the correct version. To get an ETag, call [getSyncAuthorization](getSyncAuthorization). If you don't pass the ETag in the call to `setSyncAuthorization`, then the existing authorization is overwritten indiscriminately. For more information, see [Configure the Synchronizer](https://cloud.google.com/apigee/docs/hybrid/latest/synchronizer-access). **Note**: Available to Apigee hybrid only.

      Args:
        request: (ApigeeOrganizationsSetSyncAuthorizationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SyncAuthorization) The response message.
      """
    config = self.GetMethodConfig('SetSyncAuthorization')
    return self._RunMethod(config, request, global_params=global_params)