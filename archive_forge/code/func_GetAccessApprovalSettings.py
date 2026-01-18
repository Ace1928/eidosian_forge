from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accessapproval.v1 import accessapproval_v1_messages as messages
def GetAccessApprovalSettings(self, request, global_params=None):
    """Gets the settings associated with a project, folder, or organization.

      Args:
        request: (AccessapprovalProjectsGetAccessApprovalSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessApprovalSettings) The response message.
      """
    config = self.GetMethodConfig('GetAccessApprovalSettings')
    return self._RunMethod(config, request, global_params=global_params)