from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privilegedaccessmanager.v1beta import privilegedaccessmanager_v1beta_messages as messages
def Deny(self, request, global_params=None):
    """DenyGrant is like ApproveGrant but is used for denying a Grant. This method can only be called while the Grant is in `APPROVAL_AWAITED` state. This operation cannot be undone.

      Args:
        request: (PrivilegedaccessmanagerProjectsLocationsEntitlementsGrantsDenyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Grant) The response message.
      """
    config = self.GetMethodConfig('Deny')
    return self._RunMethod(config, request, global_params=global_params)